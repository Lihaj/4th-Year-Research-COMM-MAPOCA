using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;

public class PushAgentCollab : Agent
{
    [Header("Communication Settings")]
    public int messageSize = 4; // Must match the Buffer Sensor observable size
    
    public PushBlockEnvController envController;
    private BufferSensorComponent m_BufferSensor; 
    private PushBlockSettings m_PushBlockSettings;
    private Rigidbody m_AgentRb;  // cached on initialization
    public float[] attentionWeights = new float[3];

    protected override void Awake()
    {
        base.Awake();
        m_PushBlockSettings = FindFirstObjectByType<PushBlockSettings>();
        
        // Grab the Buffer Sensor attached to this agent
        m_BufferSensor = GetComponent<BufferSensorComponent>();
    }

    public override void Initialize()
    {
        // Cache the agent rb
        m_AgentRb = GetComponent<Rigidbody>();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Pull the raw stack of messages from the controller
        if (envController != null && m_BufferSensor != null)
        {
            List<float[]> allMessages = envController.GetAllMessages();
            
            // Append each raw message into the Buffer Sensor for PyTorch
            foreach (float[] msg in allMessages)
            {
                m_BufferSensor.AppendObservation(msg);
            }
        }
    }

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
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

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
  public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);

        var continuousActions = actionBuffers.ContinuousActions;
        float[] myBroadcastMessage = new float[messageSize];
        
        // Read 0, 1, 2, 3 as the message
        for (int i = 0; i < messageSize; i++)
        {
            myBroadcastMessage[i] = continuousActions[i];
        }

        // Read 4, 5, 6 as the smuggled Attention Weights
        for (int i = 0; i < 3; i++) // 3 is the max number of agents
        {
            // We use Mathf.Clamp to ensure the opacity stays between 0.0 and 1.0
            attentionWeights[i] = Mathf.Clamp(continuousActions[messageSize + i], 0f, 1f);
        }

        if (envController != null)
        {
            envController.RegisterAgentMessage(this, myBroadcastMessage);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        var continuousActionsOut = actionsOut.ContinuousActions;

        // Default communication to 0 during manual play
        for (int i = 0; i < messageSize; i++)
        {
            continuousActionsOut[i] = 0f;
        }

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

    private void OnDrawGizmos()
    {
        // Only draw if the game is running and we have an env controller
        if (Application.isPlaying && envController != null)
        {
            for (int i = 0; i < envController.AgentsList.Count; i++)
            {
                PushAgentCollab targetAgent = envController.AgentsList[i].Agent;
                
                // Don't draw a line to ourselves
                if (targetAgent != this && targetAgent.gameObject.activeInHierarchy)
                {
                    float weight = attentionWeights[i];
                    
                    // Only draw a line if this agent is paying more than 5% attention to the target
                    if (weight > 0.05f) 
                    {
                        // Color: Green, Opacity: matches the attention weight
                        Gizmos.color = new Color(0f, 1f, 0f, weight); 
                        
                        // Draw the targeted communication beam!
                        Gizmos.DrawLine(transform.position, targetAgent.transform.position);
                        
                        // Optional: Draw a little sphere at the target to make it look like a "hit"
                        Gizmos.DrawSphere(targetAgent.transform.position + Vector3.up, 0.2f * weight);
                    }
                }
            }
        }
    }
}