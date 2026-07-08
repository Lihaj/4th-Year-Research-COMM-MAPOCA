using UnityEngine;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Lihaj.CommMAPOCA;

/// <summary>
/// The Blind Walker: can move, knows which COLOR it wants, but cannot see where
/// any goal is. The only route to the correct goal's location is the emergent
/// messages received from the Oracle watchers.
///
/// Built on the Comm-MAPOCA package: CommAgent fills the CommBuffer with incoming
/// messages, broadcasts this agent's own (unused-by-others) message, and reports
/// trainer-smuggled attention weights to the CommChannel for the gizmo lines.
/// </summary>
public class BlindWalkerAgent : CommAgent
{
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float turnSpeed = 200f;
    Rigidbody rb;

    [Header("Environment")]
    public OracleEnvController envController;

    protected override void OnCommInitialize()
    {
        rb = GetComponent<Rigidbody>();
        // The Walker is the receiver: read the attention weights the trainer smuggles
        // into spare continuous action slots so CommChannel can color the flow lines.
        readSmuggledAttention = true;
    }

    protected override void CollectAgentObservations(VectorSensor sensor)
    {
        // Shared 9-float layout (role flag disambiguates Oracle/Walker):
        //   [0]    role flag (1 = Walker)
        //   [1..2] heading  : facing direction in the ENVIRONMENT frame (x,z)
        //   [3..4] velocity : body-frame velocity / moveSpeed
        //   [5..6] position : own local position (x,z) / spawnArea
        //   [7..8] wanted-color one-hot (the color of goal to reach)
        // ALL environment-relative, so parallel training arenas stay consistent.
        // Heading stays essential even with position: tank controls (forward/turn)
        // are relative to facing, so the Walker needs its heading to convert a
        // goal-position message into "turn left / go forward".
        sensor.AddObservation(1f); // role flag: Walker

        Vector3 fwd = transform.localRotation * Vector3.forward;
        sensor.AddObservation(fwd.x);
        sensor.AddObservation(fwd.z);

        Vector3 localVel = Vector3.zero;
        if (rb != null)
            localVel = transform.InverseTransformDirection(rb.linearVelocity) / Mathf.Max(moveSpeed, 0.0001f);
        sensor.AddObservation(localVel.x);
        sensor.AddObservation(localVel.z);

        float posNorm = (envController != null) ? 1f / Mathf.Max(envController.spawnArea, 0.0001f) : 0.125f;
        sensor.AddObservation(transform.localPosition.x * posNorm);
        sensor.AddObservation(transform.localPosition.z * posNorm);

        if (envController != null && envController.currentTargetID == 0)
        { sensor.AddObservation(1f); sensor.AddObservation(0f); } // hunt Red
        else if (envController != null)
        { sensor.AddObservation(0f); sensor.AddObservation(1f); } // hunt Blue
        else
        { sensor.AddObservation(0f); sensor.AddObservation(0f); }

        // Incoming messages are appended to the CommBuffer automatically by CommAgent.
    }

    protected override void OnCommActionReceived(ActionBuffers actionBuffers)
    {
        // Movement (1 discrete branch, size 5). The message (continuous [0..3]) and
        // the smuggled attention weights (continuous [4..6]) are consumed by CommAgent.
        int moveAction = actionBuffers.DiscreteActions[0];
        Vector3 moveDir = Vector3.zero;
        float rotationInput = 0f;

        if (moveAction == 1) moveDir = transform.forward;
        if (moveAction == 2) moveDir = -transform.forward;
        if (moveAction == 3) rotationInput = -1f;
        if (moveAction == 4) rotationInput = 1f;

        rb.linearVelocity = moveDir * moveSpeed;
        transform.Rotate(Vector3.up, rotationInput * turnSpeed * Time.fixedDeltaTime);
    }

    void OnTriggerEnter(Collider other)
    {
        if (envController == null) return;

        if (other.CompareTag("RedGoal") && envController.currentTargetID == 0)
        {
            envController.ResolveEpisode(true); // correct goal!
        }
        else if (other.CompareTag("BlueGoal") && envController.currentTargetID == 1)
        {
            envController.ResolveEpisode(true); // correct goal!
        }
        else if (other.CompareTag("RedGoal") || other.CompareTag("BlueGoal"))
        {
            // Wrong color: small penalty, episode continues (walker can keep searching)
            envController.ApplyMinorPenalty(-0.1f);
        }
    }

    // Communication flow gizmos are drawn by CommChannel (attention-colored lines);
    // the old per-agent OnDrawGizmos laser code is no longer needed.
}
