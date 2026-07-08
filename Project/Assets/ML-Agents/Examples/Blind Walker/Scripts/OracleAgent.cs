using UnityEngine;
using Unity.MLAgents.Sensors;
using Lihaj.CommMAPOCA;

/// <summary>
/// Oracle "watcher": stationary agent that can see where its colored goal and the
/// Walker are, and must LEARN to communicate that to the color-blind Walker.
///
/// Built on the Comm-MAPOCA package: CommAgent broadcasts the first 4 continuous
/// actions as this agent's message (tanh-bounded) and fills the CommBuffer sensor
/// automatically -- no communication plumbing needed here. Messages are fully
/// emergent (no hand-coded content).
/// </summary>
public class OracleAgent : CommAgent
{
    [Header("Oracle Knowledge")]
    [Tooltip("Drag the specific colored goal this Oracle cares about here")]
    public Transform myTargetGoal;

    [Tooltip("Drag the Blind Walker here")]
    public Transform blindWalker;

    [Header("Color Identity")]
    [Tooltip("0 = Red, 1 = Blue. Auto-detected from the assigned goal's tag at startup.")]
    public int myGoalColorID = 0;

    protected override void OnCommInitialize()
    {
        // Detect which color this Oracle watches from the tag of its assigned goal
        // (RedGoal/BlueGoal), so it can learn a color-distinguishable message the
        // Walker's query can target. Falls back to the Inspector value if untagged.
        if (myTargetGoal != null)
        {
            if (myTargetGoal.CompareTag("RedGoal")) myGoalColorID = 0;
            else if (myTargetGoal.CompareTag("BlueGoal")) myGoalColorID = 1;
        }
        Debug.Log($"[Oracle {name}] watching color: {(myGoalColorID == 0 ? "RED" : "BLUE")}");
    }

    protected override void CollectAgentObservations(VectorSensor sensor)
    {
        // Oracle and Walker share one Behavior Name/network, so both must produce
        // the same-shaped 9-float vector. Layout:
        //   [0]    role flag (0 = Oracle, 1 = Walker)
        //   [1..6] Oracle: goal position + walker position (env-local)
        //   [7..8] color one-hot: the color THIS Oracle watches (same slots the
        //          Walker uses for its wanted color; role flag disambiguates)
        sensor.AddObservation(0f); // role flag: Oracle

        sensor.AddObservation(myTargetGoal != null ? myTargetGoal.localPosition : Vector3.zero);
        sensor.AddObservation(blindWalker != null ? blindWalker.localPosition : Vector3.zero);

        if (myGoalColorID == 0) { sensor.AddObservation(1f); sensor.AddObservation(0f); } // Red
        else                    { sensor.AddObservation(0f); sensor.AddObservation(1f); } // Blue
    }

    // The outgoing message is handled entirely by CommAgent (learned policy output,
    // tanh-bounded). The Oracle takes no physical actions.
}
