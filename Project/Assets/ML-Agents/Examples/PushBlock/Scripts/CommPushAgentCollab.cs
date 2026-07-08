using UnityEngine;
using Unity.MLAgents.Actuators;
using Lihaj.CommMAPOCA;

/// <summary>
/// Communication-enabled cooperative PushBlock agent (Comm-MAPOCA benchmark variant).
///
/// Identical push behavior to the stock PushAgentCollab, but built on the
/// com.lihaj.comm-mapoca package: each agent broadcasts a learned 4-float message
/// (first 4 continuous actions, tanh-bounded) through the arena's CommChannel and
/// receives its teammates' messages via the auto-configured CommBuffer sensor
/// (All-to-All topology). Continuous action slots after the message carry
/// trainer-smuggled attention weights for the gizmo lines.
///
/// Behavior parameters for the comm variant: same sensors as stock, PLUS
/// 6 continuous actions (4 message + 2 attention-gizmo spares) alongside the
/// stock 1 discrete branch of size 7. Use a NEW behavior name (e.g.
/// "CommPushBlock") so it never collides with the stock baseline.
/// </summary>
public class CommPushAgentCollab : CommAgent
{
    private PushBlockSettings m_PushBlockSettings;
    private Rigidbody m_AgentRb;

    protected override void Awake()
    {
        base.Awake();
        m_PushBlockSettings = FindFirstObjectByType<PushBlockSettings>();
    }

    protected override void OnCommInitialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        // Every agent is a receiver here (All-to-All), so let each report its
        // attention weights for the CommChannel gizmo coloring.
        readSmuggledAttention = true;
    }

    /// <summary>Moves the agent according to the selected action (stock mapping).</summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 1:
                dirToGo = transform.forward * 1f;
                break;
            case 2:
                dirToGo = transform.forward * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                rotateDir = transform.up * -1f;
                break;
            case 5:
                dirToGo = transform.right * -0.75f;
                break;
            case 6:
                dirToGo = transform.right * 0.75f;
                break;
        }
        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_PushBlockSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    protected override void OnCommActionReceived(ActionBuffers actionBuffers)
    {
        // The message (continuous [0..3]) and attention spares ([4..5]) are consumed
        // by CommAgent; this agent only handles movement.
        MoveAgent(actionBuffers.DiscreteActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 4;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }
}
