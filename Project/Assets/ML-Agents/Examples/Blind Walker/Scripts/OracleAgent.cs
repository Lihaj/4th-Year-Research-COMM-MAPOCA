using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class OracleAgent : Agent
{
    [Header("Oracle Knowledge")]
    [Tooltip("Drag the specific colored goal this Oracle cares about here")]
    public Transform myTargetGoal; 
    
    [Tooltip("Drag the Blind Walker here")]
    public Transform blindWalker;

    [Header("Communication")]
    public int messageSize = 4;
    public OracleEnvController envController;

    [Header("Color Identity")]
    [Tooltip("0 = Red, 1 = Blue. Auto-detected from the assigned goal's tag at startup.")]
    public int myGoalColorID = 0;

    [Header("Sanity-Check Mode")]
    [Tooltip("If ON, this Oracle broadcasts the TRUE goal info instead of a learned message: " +
             "[goalX/spawnArea, goalZ/spawnArea, color(+1 red / -1 blue), 0]. Proves the " +
             "message -> attention -> navigation pipeline works with known-good comms. " +
             "Turn OFF on all Oracles to return to emergent (learned) messages.")]
    public bool useHandCodedMessage = true;

    public override void Initialize()
    {
        // Figure out which color goal this Oracle watches, so it can broadcast a
        // color-tagged message the Walker's query can match against. We read it from
        // the tag of the goal you already dragged into 'myTargetGoal' (RedGoal/BlueGoal),
        // so there is no extra Inspector setup. If the goal isn't tagged, we fall back
        // to whatever myGoalColorID is set to in the Inspector.
        if (myTargetGoal != null)
        {
            if (myTargetGoal.CompareTag("RedGoal")) myGoalColorID = 0;
            else if (myTargetGoal.CompareTag("BlueGoal")) myGoalColorID = 1;
        }
        // One-time startup confirmation -- safe to comment out once you've verified it.
        Debug.Log($"[Oracle {name}] watching color: {(myGoalColorID == 0 ? "RED" : "BLUE")}");
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // COMM-MAPOCA: Oracle and Walker now share one Behavior Name/network, so both
        // must produce the same-shaped observation vector. Layout (9 floats total):
        //   [0]      role flag (0 = Oracle, 1 = Walker)
        //   [1..6]   Oracle-only info (goal pos + walker pos) -- zero when role=Walker
        //   [7..8]   Walker-only info (one-hot target color)  -- zero when role=Oracle
        sensor.AddObservation(0f); // role flag: Oracle

        // 1. Tell the neural network exactly where the goal is (3 floats)
        if (myTargetGoal != null)
            sensor.AddObservation(myTargetGoal.localPosition);
        else
            sensor.AddObservation(Vector3.zero);

        // 2. Tell the neural network exactly where the Walker is (3 floats)
        if (blindWalker != null)
            sensor.AddObservation(blindWalker.localPosition);
        else
            sensor.AddObservation(Vector3.zero);

        // Color identity of the goal THIS Oracle watches (one-hot): [1,0]=Red, [0,1]=Blue.
        // Occupies the SAME two slots the Walker uses for its wanted-color, so the shared
        // network learns a single color feature at a consistent position. The role flag at
        // [0] tells the network to read this as "the color I watch" (vs. the Walker's
        // "the color I want"). This is what lets the Oracle put a color signature into its
        // message so the Walker's query can target the right Oracle.
        if (myGoalColorID == 0) { sensor.AddObservation(1f); sensor.AddObservation(0f); } // Red
        else                    { sensor.AddObservation(0f); sensor.AddObservation(1f); } // Blue

        // Total Vector Space Size = 9
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // The Oracle cannot move (0 discrete actions), it only speaks (4 continuous actions).
        var continuousActions = actionBuffers.ContinuousActions;
        float[] myBroadcastMessage = new float[messageSize];

        if (useHandCodedMessage && myTargetGoal != null && envController != null)
        {
            // SANITY-CHECK MODE: broadcast the TRUE goal info instead of a learned message,
            // to prove the pipeline works with known-good communication. Values are LOCAL
            // (environment-relative, from localPosition) and normalized, so they are valid
            // across every parallel training environment.
            float norm = 1f / Mathf.Max(envController.spawnArea, 0.0001f);
            myBroadcastMessage[0] = Mathf.Clamp(myTargetGoal.localPosition.x * norm, -1f, 1f); // goal X
            myBroadcastMessage[1] = Mathf.Clamp(myTargetGoal.localPosition.z * norm, -1f, 1f); // goal Z
            myBroadcastMessage[2] = (myGoalColorID == 0) ? 1f : -1f;                            // +1 red, -1 blue
            myBroadcastMessage[3] = 0f;                                                          // spare (kept 4-wide)
        }
        else
        {
            // EMERGENT MODE: the message is whatever the policy chooses, tanh-bounded to [-1,1].
            // Raw continuous outputs are unbounded and would otherwise grow without limit and
            // destabilize the (unnormalized) critic.
            for (int i = 0; i < messageSize; i++)
            {
                myBroadcastMessage[i] = (float)System.Math.Tanh(continuousActions[i]);
            }
        }

        // Send the message to the Controller's bulletin board
        if (envController != null)
        {
            envController.RegisterAgentMessage(this, myBroadcastMessage);
        }
    }
}