using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Two-float vector sensor giving the agent the goal zone's offset in the
/// agent's LOCAL frame (x = right, z = forward), normalized to [-1, 1].
///
/// Purpose (v6 minimal experiment): under 7x7 partial observability a
/// memoryless policy cannot locate an unseen goal, making "which way do I
/// push?" unanswerable and inviting reward farming. Providing the goal
/// direction removes that unsolvable sub-problem while keeping the
/// information asymmetry under study intact: block and teammate positions
/// remain visible only within each agent's local grid view. Attach to BOTH
/// the comm-arm and baseline-arm agents so the observation spec stays
/// identical across arms.
///
/// The goal reference is auto-found from the parent arena's
/// CommPushBlockEnvController.goalZone; leave the field empty in prefabs.
/// </summary>
public class GoalDirectionSensorComponent : SensorComponent
{
    [Tooltip("The goal transform. Leave null to auto-find the parent arena controller's goalZone.")]
    public Transform goal;

    [Tooltip("World units mapped to observation value 1.0. Arena half-diagonal ~12.5 works for the 25x25 PushBlock arena.")]
    public float normalizationScale = 12.5f;

    public override ISensor[] CreateSensors()
    {
        var target = goal;
        if (target == null)
        {
            var ctrl = GetComponentInParent<CommPushBlockEnvController>();
            if (ctrl != null)
                target = ctrl.goalZone;
        }
        if (target == null)
            Debug.LogError($"[GoalDirectionSensor] No goal found for {name} -- assign one or ensure a parent CommPushBlockEnvController has goalZone set.", this);

        return new ISensor[] { new GoalDirectionSensor(transform, target, normalizationScale) };
    }
}

public class GoalDirectionSensor : ISensor
{
    readonly Transform m_Agent;
    readonly Transform m_Goal;
    readonly float m_Scale;
    readonly ObservationSpec m_Spec = ObservationSpec.Vector(2);

    public GoalDirectionSensor(Transform agent, Transform goal, float scale)
    {
        m_Agent = agent;
        m_Goal = goal;
        m_Scale = Mathf.Max(scale, 0.0001f);
    }

    public int Write(ObservationWriter writer)
    {
        if (m_Goal == null)
        {
            writer[0] = 0f;
            writer[1] = 0f;
            return 2;
        }
        // Egocentric: the offset rotates with the agent, so the values answer
        // "which way is the goal RELATIVE TO MY FACING" -- directly usable
        // with the tank-style controls (forward/turn).
        var rel = m_Goal.position - m_Agent.position;
        var local = m_Agent.InverseTransformDirection(rel);
        writer[0] = Mathf.Clamp(local.x / m_Scale, -1f, 1f);
        writer[1] = Mathf.Clamp(local.z / m_Scale, -1f, 1f);
        return 2;
    }

    public ObservationSpec GetObservationSpec() => m_Spec;
    public byte[] GetCompressedObservation() => null;
    public CompressionSpec GetCompressionSpec() => CompressionSpec.Default();
    public void Update() { }
    public void Reset() { }
    public string GetName() => "GoalDirection";
}
