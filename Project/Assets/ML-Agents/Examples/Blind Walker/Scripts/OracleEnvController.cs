using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Lihaj.CommMAPOCA;

/// <summary>
/// Environment controller for the Blind Walker task: manages the MA-POCA group,
/// team rewards (win / timeout / "getting warmer" shaping), and arena resets.
///
/// Communication is handled by the Comm-MAPOCA package's CommChannel component
/// (message board, topology, presence flags, gizmos) -- this controller only
/// clears the board on full resets.
/// </summary>
public class OracleEnvController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public Agent Agent;
        [HideInInspector] public Vector3 StartingPos;
        [HideInInspector] public Quaternion StartingRot;
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

    [Header("Communication")]
    [Tooltip("The Comm-MAPOCA channel for this arena. Auto-found on this GameObject if left null.")]
    public CommChannel commChannel;

    [Header("Training Settings")]
    [Tooltip("Maximum physics steps before the episode times out.")]
    public int maxEnvironmentSteps = 2500;

    [Tooltip("How far from the center agents and goals can spawn.")]
    public float spawnArea = 8f;

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
             "0 = always Red, 1 = always Blue.")]
    public int forceTargetColor = -1;

    private int resetTimer;

    // MA-POCA requires a group controller to handle team rewards
    private SimpleMultiAgentGroup m_AgentGroup;

    [HideInInspector]
    public int currentTargetID = 0; // 0 = Red, 1 = Blue

    void Start()
    {
        if (commChannel == null)
            commChannel = GetComponent<CommChannel>();

        m_AgentGroup = new SimpleMultiAgentGroup();
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
        // Existential pressure: tiny negative reward every physics step
        m_AgentGroup.AddGroupReward(-1.0f / maxEnvironmentSteps);

        // "Getting warmer" shaping (potential-based): reward the team for the Walker
        // reducing its distance to the CORRECT-color goal. The Walker is blind to
        // positions, so the only way to earn this is by using the Watchers' messages.
        // Reward is never an observation, so it cannot be gamed directly.
        if (warmerRewardScale != 0f && blindWalker != null)
        {
            float curDist = Vector3.Distance(
                blindWalker.transform.position, CorrectGoalTransform().position);
            m_AgentGroup.AddGroupReward((m_PrevDistanceToGoal - curDist) * warmerRewardScale);
            m_PrevDistanceToGoal = curDist;
        }

        // Timeout
        resetTimer += 1;
        if (resetTimer >= maxEnvironmentSteps && maxEnvironmentSteps > 0)
        {
            m_AgentGroup.AddGroupReward(-1.0f);
            m_AgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }
    }

    /// <summary>The goal the Walker is currently supposed to reach.</summary>
    private Transform CorrectGoalTransform()
    {
        return (currentTargetID == 0) ? goalRed.transform : goalBlue.transform;
    }

    /// <summary>Resets the arena, shuffles goal positions, and picks a new target color.</summary>
    public void ResetScene()
    {
        resetTimer = 0;

        // Fresh episode = fresh message board (per-agent slots are also cleared by
        // CommAgent.OnEpisodeBegin; this handles the full-arena reset path).
        if (commChannel != null)
            commChannel.ClearBoard();

        // Target color: forced for curriculum (>= 0) or random each episode
        currentTargetID = (forceTargetColor >= 0) ? forceTargetColor : Random.Range(0, 2);

        if (targetIndicator != null)
            targetIndicator.material.color = (currentTargetID == 0) ? Color.red : Color.blue;

        // Randomize positions
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

        // Seed the shaping baseline so the first "warmer" delta is ~0
        m_PrevDistanceToGoal = Vector3.Distance(
            blindWalker.transform.position, CorrectGoalTransform().position);
    }

    /// <summary>Called by the Walker when it touches a goal sphere.</summary>
    public void ResolveEpisode(bool success)
    {
        m_AgentGroup.AddGroupReward(success ? winReward : -1.0f);
        m_AgentGroup.EndGroupEpisode();
        ResetScene();
    }

    /// <summary>Small team punishment WITHOUT ending the episode (wrong-color goal).</summary>
    public void ApplyMinorPenalty(float amount)
    {
        if (m_AgentGroup != null)
            m_AgentGroup.AddGroupReward(amount);
    }
}
