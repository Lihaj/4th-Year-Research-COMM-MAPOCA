using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class BlindWalkerAgent : Agent
{
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float turnSpeed = 200f;
    private Rigidbody rb;

    [Header("Communication")]
    public OracleEnvController envController;
    public int messageSize = 4;
    
    // Stores the smuggled attention weights from PyTorch for the Gizmo lasers
    private float[] attentionWeights = new float[3]; 
    private BufferSensorComponent m_BufferSensor;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        m_BufferSensor = GetComponent<BufferSensorComponent>();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // COMM-MAPOCA: Oracle and Walker share one Behavior Name/network, so both must produce
        // the same-shaped 9-float vector (the role flag at [0] tells the net which is which).
        // Walker layout: [0] role=1 | [1..2] heading | [3..4] velocity | [5..6] own position |
        //                [7..8] target color one-hot.  ALL values are environment-relative.
        sensor.AddObservation(1f); // role flag: Walker

        // COMM-MAPOCA proprioception (ALL environment-relative via local space, so parallel
        // training environments stay consistent). Fills slots [1..6]:
        //   [1..2] heading  : forward direction in the ENVIRONMENT's frame (x,z)
        //   [3..4] velocity : body-frame velocity / moveSpeed (already orientation-relative)
        //   [5..6] position : own local position (x,z) / spawnArea, relative to the environment
        // Heading stays essential even with position: the Walker moves with tank controls
        // (forward/turn are relative to its facing), so it needs its facing to turn a goal-
        // position message into "turn left / go forward". Position = WHERE the goal is;
        // heading = WHICH WAY the Walker points.
        Vector3 fwd = transform.localRotation * Vector3.forward; // facing, in the environment frame
        sensor.AddObservation(fwd.x);
        sensor.AddObservation(fwd.z);

        Vector3 localVel = Vector3.zero;
        if (rb != null)
            localVel = transform.InverseTransformDirection(rb.linearVelocity) / Mathf.Max(moveSpeed, 0.0001f);
        sensor.AddObservation(localVel.x);
        sensor.AddObservation(localVel.z);

        float posNorm = (envController != null) ? 1f / Mathf.Max(envController.spawnArea, 0.0001f) : 0.125f;
        sensor.AddObservation(transform.localPosition.x * posNorm); // own X (env-local, normalized)
        sensor.AddObservation(transform.localPosition.z * posNorm); // own Z (env-local, normalized)

        // We use "One-Hot Encoding" to tell the agent what color to hunt for.
        if (envController != null)
        {
            if (envController.currentTargetID == 0) // Hunt Red
            {
                sensor.AddObservation(1f);
                sensor.AddObservation(0f);
            }
            else // Hunt Blue
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(1f);
            }

          if (m_BufferSensor != null)
            {
                // GetAllMessages(this) returns fresh, correctly-sized arrays
                // (4 content floats + 1 explicit presence flag), safe to append directly.
                // Passing `this` excludes our OWN message from the buffer -- required for
                // the critic's counterfactual baseline (see OracleEnvController comment).
                List<float[]> allMessages = envController.GetAllMessages(this);
                foreach (float[] msg in allMessages)
                {
                    m_BufferSensor.AppendObservation(msg);
                }
            }
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // --- 1. MOVEMENT (1 Discrete Branch, Size 5) ---
        int moveAction = actionBuffers.DiscreteActions[0];
        Vector3 moveDir = Vector3.zero;
        float rotationInput = 0f;

        if (moveAction == 1) moveDir = transform.forward;       // Move Forward
        if (moveAction == 2) moveDir = -transform.forward;      // Move Backward
        if (moveAction == 3) rotationInput = -1f;               // Turn Left
        if (moveAction == 4) rotationInput = 1f;                // Turn Right

        rb.linearVelocity = moveDir * moveSpeed;
        transform.Rotate(Vector3.up, rotationInput * turnSpeed * Time.fixedDeltaTime);

        // --- 2. COMMUNICATION & GIZMOS (7 Continuous Actions) ---
        var continuousActions = actionBuffers.ContinuousActions;
        float[] myBroadcastMessage = new float[messageSize];

        // Slots 0, 1, 2, 3: The actual message.
        // COMM-MAPOCA: squash to [-1, 1] with tanh before broadcasting -- raw continuous
        // outputs are unbounded and would grow without limit, destabilizing the (unnormalized)
        // critic that now consumes these messages. tanh keeps the range bounded but expressive.
        for (int i = 0; i < messageSize; i++)
        {
            myBroadcastMessage[i] = (float)System.Math.Tanh(continuousActions[i]);
        }

        // Slots 4, 5, 6: The smuggled Attention Weights
        for (int i = 0; i < 3; i++)
        {
            attentionWeights[i] = Mathf.Clamp(continuousActions[messageSize + i], 0f, 1f);
        }

        // COMM-MAPOCA TEMP: commented out to keep console clean during the groupmate-visibility diagnostic
        // Debug.Log($"Raw Attention -> A: {attentionWeights[0]:F2} | B: {attentionWeights[1]:F2} | Self: {attentionWeights[2]:F2}");

        // Send our message to the bulletin board
        if (envController != null)
        {
            envController.RegisterAgentMessage(this, myBroadcastMessage);
        }
    }

    // --- 3. GOAL DETECTION ---
private void OnTriggerEnter(Collider other)
    {
        if (envController == null) return;

        // 1. Did we hit the Red goal while hunting Red?
        if (other.CompareTag("RedGoal") && envController.currentTargetID == 0)
        {
            envController.ResolveEpisode(true); // Success!
        }
        // 2. Did we hit the Blue goal while hunting Blue?
        else if (other.CompareTag("BlueGoal") && envController.currentTargetID == 1)
        {
            envController.ResolveEpisode(true); // Success!
        }
        // 3. Did we hit the WRONG goal?
        else if (other.CompareTag("RedGoal") || other.CompareTag("BlueGoal"))
        {
            // Give a small penalty (-0.1f) but DO NOT end the episode!
            // The Walker passes through and can keep searching.
            envController.ApplyMinorPenalty(-0.1f); 
        }
    }

    // --- 4. THE TARMAC LASER BEAMS ---
    private void OnDrawGizmos()
    {
        if (Application.isPlaying && envController != null)
        {
            for (int i = 0; i < envController.AgentsList.Count; i++)
            {
                Agent targetAgent = envController.AgentsList[i].Agent;
                
                // Safely check that the agent exists before drawing!
                if (targetAgent != null && targetAgent != this && targetAgent.gameObject.activeInHierarchy)
                {
                    float weight = attentionWeights[i];
                    
                    // Only draw if attention is above 5% to reduce visual clutter
                    if (weight > 0.05f) 
                    {
                        // --- THE COLOR GRADIENT ---
                        // Color.Lerp blends from Red (0.0 weight) to Green (1.0 weight).
                        // We also keep the line 100% solid (opacity 1f) so the colors pop!
                        Color gradientColor = Color.Lerp(Color.red, Color.green, weight);
                        gradientColor.a = 1f; // Force full opacity
                        
                        Gizmos.color = gradientColor;
                        // --------------------------
                        
                        Gizmos.DrawLine(transform.position, targetAgent.transform.position);
                        
                        // Make the sphere at the end scale with the weight so high-attention targets have larger markers
                        Gizmos.DrawSphere(targetAgent.transform.position, 0.5f * weight);
                    }
                }
            }
        }
    }
}