using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class OracleEnvController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public Agent Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
    }

    [Header("Agent Roster")]
    [Tooltip("Add all 3 agents (Watcher A, Watcher B, Walker) to this list in a strict order.")]
    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();

    [Header("Environment Objects")]
    public GameObject goalRed;
    public GameObject goalBlue;
    public GameObject blindWalker;

    [Tooltip("The floating sphere that shows the current target color")]
    public MeshRenderer targetIndicator;

    [Header("Training Settings")]
    [Tooltip("Maximum physics steps before the episode times out.")]
    public int maxEnvironmentSteps = 2500; // ~50 seconds of physics time

    [Tooltip("How far from the center agents and goals can spawn.")]
    public float spawnArea = 8f; // <--- ADD THIS LINE

    [Header("Reward Settings")]
    [Tooltip("Team reward when the Walker reaches the CORRECT-color goal.")]
    public float winReward = 3f;

    [Tooltip("Scale for the potential-based 'getting warmer' shaping: each physics step, " +
             "reward += (prevDistance - currentDistance) * this, measured to the correct goal. " +
             "Set to 0 to disable shaping.")]
    public float warmerRewardScale = 0.1f;

    // Tracks the Walker's distance to the current correct goal, for potential-based shaping.
    private float m_PrevDistanceToGoal;

    [Header("Curriculum")]
    [Tooltip("Force the target color every episode. -1 = random (normal, full task). " +
             "0 = always Red, 1 = always Blue. Use a fixed color to master navigation first, " +
             "then set back to -1 to restore the full color-selection task.")]
    public int forceTargetColor = 0;

    private int resetTimer;

    // MA-POCA requires a group controller to handle team rewards
    private SimpleMultiAgentGroup m_AgentGroup;

    // The central bulletin board for TarMAC messages (Key = Agent, Value = 4-float message)
    private Dictionary<Agent, float[]> currentMessages = new Dictionary<Agent, float[]>();

    [HideInInspector]
    public int currentTargetID = 0; // 0 = Red, 1 = Blue

    void Start()
    {
        m_AgentGroup = new SimpleMultiAgentGroup();
        
        // Register all agents to the MA-POCA team
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.localPosition;
            item.StartingRot = item.Agent.transform.localRotation;
            m_AgentGroup.RegisterAgent(item.Agent);
        }
        ResetScene();
    }

    void FixedUpdate()
    {
        // --- EXISTENTIAL PENALTY ---
        // Give a microscopic negative reward every single physics step.
        // If maxSteps is 5000, this applies -0.0002 per step. 
        m_AgentGroup.AddGroupReward(-1.0f / maxEnvironmentSteps);

        // --- "GETTING WARMER" SHAPING (potential-based) ---
        // Reward the team for the Walker reducing its distance to the CORRECT-color goal.
        // The Walker is blind to positions, so the ONLY way it can get closer to the right
        // goal is by correctly using the Watchers' messages -- this densely rewards working
        // communication, and because reward is never an observation the Walker can't "cheat"
        // by sensing warmth directly. The delta-distance (potential-based) form leaves the
        // optimal policy unchanged.
        if (warmerRewardScale != 0f && blindWalker != null)
        {
            float curDist = Vector3.Distance(
                blindWalker.transform.position, CorrectGoalTransform().position);
            m_AgentGroup.AddGroupReward((m_PrevDistanceToGoal - curDist) * warmerRewardScale);
            m_PrevDistanceToGoal = curDist;
        }

        // --- THE DOOMSDAY CLOCK ---
        resetTimer += 1;
        if (resetTimer >= maxEnvironmentSteps && maxEnvironmentSteps > 0)
        {
            m_AgentGroup.AddGroupReward(-1.0f);
            m_AgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }
    }

    /// <summary>
    /// Senders call this to post their message to the board.
    /// </summary>
    public void RegisterAgentMessage(Agent sender, float[] message)
    {
        currentMessages[sender] = message;
    }

    /// <summary>
    /// The Receiver calls this to read the board in a strict, unchanging order.
    /// Each entry is 5 floats: [0..3] = the actual message content, [4] = presence flag
    /// (1.0 if this agent has actually broadcast a message this episode, 0.0 otherwise).
    /// Using an explicit flag (rather than inferring "real vs. placeholder" from the
    /// message's magnitude) avoids ever mistaking a genuine all-zero/never-spoken slot
    /// for real content -- see the old "Anti-Ghost Tag" hack this replaces.
    ///
    /// COMM-MAPOCA: the requester's OWN message is EXCLUDED. Reason: the requester's
    /// CommBuffer is used by the Python critic as target-agent j's "received messages"
    /// in the counterfactual baseline, and j's own sent message m_j is one of the
    /// quantities the baseline marginalizes out -- including it would leak m_j into
    /// its own counterfactual and corrupt the advantage. (The requester loses TarMAC
    /// self-attention, which carries no information it doesn't already have.)
    /// NOTE for the gizmos: keep the Walker LAST in AgentsList so the smuggled
    /// attention weights [0..1] still line up with Watcher A / Watcher B.
    /// </summary>
    public List<float[]> GetAllMessages(Agent requester)
    {
        List<float[]> activeMessages = new List<float[]>();
        foreach (var info in AgentsList)
        {
            if (info.Agent == requester) continue; // self-exclusion (see summary above)
            if (info.Agent != null && info.Agent.gameObject.activeInHierarchy)
            {
                float[] taggedMessage = new float[5]; // 4 content floats + 1 presence flag
                if (currentMessages.ContainsKey(info.Agent))
                {
                    System.Array.Copy(currentMessages[info.Agent], taggedMessage, 4);
                    taggedMessage[4] = 1f; // Has spoken this episode
                }
                // else: taggedMessage stays all zeros, including the presence flag (0f),
                // correctly marking this as "hasn't spoken yet" rather than real content.
                activeMessages.Add(taggedMessage);
            }
        }
        return activeMessages;
    }

    /// <summary>
    /// Returns the goal the Walker is currently supposed to reach (based on currentTargetID).
    /// Used by the "getting warmer" shaping to measure progress toward the right color.
    /// </summary>
    private Transform CorrectGoalTransform()
    {
        return (currentTargetID == 0) ? goalRed.transform : goalBlue.transform;
    }

    /// <summary>
    /// Resets the arena, shuffles goal positions, and picks a new target color.
    /// </summary>
    public void ResetScene()
    {
        // Reset the timer for the new episode
        resetTimer = 0;
        currentMessages.Clear();

        // Target color: forced to a fixed color for curriculum (forceTargetColor >= 0),
        // otherwise random each episode (0 = Red, 1 = Blue).
        currentTargetID = (forceTargetColor >= 0) ? forceTargetColor : Random.Range(0, 2);

        // --- INDICATOR LIGHT LOGIC ---
        if (targetIndicator != null)
        {
            if (currentTargetID == 0) // Target is Red
            {
                targetIndicator.material.color = Color.red;
            }
            else // Target is Blue
            {
                targetIndicator.material.color = Color.blue;
            }
        }

        // Randomize positions (Assuming a 20x20 floor, bounds are -8 to 8)
        goalRed.transform.localPosition = new Vector3(Random.Range(-spawnArea, spawnArea), 0.5f, Random.Range(-spawnArea, spawnArea));
        goalBlue.transform.localPosition = new Vector3(Random.Range(-spawnArea, spawnArea), 0.5f, Random.Range(-spawnArea, spawnArea));
        blindWalker.transform.localPosition = new Vector3(Random.Range(-spawnArea, spawnArea), 1f, Random.Range(-spawnArea, spawnArea));

        // Kill any leftover momentum on the walker
        Rigidbody rb = blindWalker.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        // Seed the shaping baseline for the new episode so the first "getting warmer"
        // delta is ~0 (no spurious reward spike across the reset boundary).
        m_PrevDistanceToGoal = Vector3.Distance(
            blindWalker.transform.position, CorrectGoalTransform().position);
    }

    /// <summary>
    /// Called by the Walker when it touches a sphere.
    /// </summary>
    public void ResolveEpisode(bool success)
    {
        if (success)
        {
            m_AgentGroup.AddGroupReward(winReward); // The team wins!
        }
        else
        {
            m_AgentGroup.AddGroupReward(-1.0f); // The team fails!
        }
        
        m_AgentGroup.EndGroupEpisode();
        ResetScene();
    }

    /// <summary>
    /// Applies a small punishment to the team WITHOUT ending the episode.
    /// </summary>
    public void ApplyMinorPenalty(float amount)
    {
        if (m_AgentGroup != null)
        {
            m_AgentGroup.AddGroupReward(amount);
        }
    }
}