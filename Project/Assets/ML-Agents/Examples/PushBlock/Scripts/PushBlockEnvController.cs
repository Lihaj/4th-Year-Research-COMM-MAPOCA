using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class PushBlockEnvController : MonoBehaviour
{
    [Header("Communication Routing")]
    // Stores the raw message from every agent at time t
    private Dictionary<PushAgentCollab, float[]> currentMessages = new Dictionary<PushAgentCollab, float[]>();

    [System.Serializable]
    public class PlayerInfo
    {
        public PushAgentCollab Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
    }

    [System.Serializable]
    public class BlockInfo
    {
        public Transform T;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
    }

    [Header("Max Environment Steps")] 
    public int MaxEnvironmentSteps = 25000;

    [HideInInspector]
    public Bounds areaBounds;
    public GameObject ground;
    public GameObject area;

    Material m_GroundMaterial; 
    Renderer m_GroundRenderer;

    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();
    public List<BlockInfo> BlocksList = new List<BlockInfo>();

    public bool UseRandomAgentRotation = true;
    public bool UseRandomAgentPosition = true;
    public bool UseRandomBlockRotation = true;
    public bool UseRandomBlockPosition = true;
    private PushBlockSettings m_PushBlockSettings;

    private int m_NumberOfRemainingBlocks;
    private SimpleMultiAgentGroup m_AgentGroup;
    private int m_ResetTimer;

    /// <summary>
    /// Called by each agent to post their message to the board.
    /// </summary>
    public void RegisterAgentMessage(PushAgentCollab agent, float[] message)
    {
        if (!currentMessages.ContainsKey(agent))
        {
            currentMessages.Add(agent, message);
        }
        else
        {
            currentMessages[agent] = message;
        }
    }

    /// <summary>
    /// Called by each agent to grab the raw stack of all messages.
    /// </summary>
   public List<float[]> GetAllMessages()
    {
        List<float[]> activeMessages = new List<float[]>();
        
        // Loop through the strict, unchanging list of agents
        foreach (var playerInfo in AgentsList) 
        {
            PushAgentCollab agent = playerInfo.Agent;
            if (agent != null && agent.gameObject.activeInHierarchy)
            {
                if (currentMessages.ContainsKey(agent))
                {
                    activeMessages.Add(currentMessages[agent]);
                }
                else
                {
                    // Fallback if an agent hasn't spoken yet
                    activeMessages.Add(new float[4]); 
                }
            }
        }
        return activeMessages;
    }

    void Start()
    {
        areaBounds = ground.GetComponent<Collider>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        m_PushBlockSettings = FindFirstObjectByType<PushBlockSettings>();
        
        foreach (var item in BlocksList)
        {
            item.StartingPos = item.T.transform.position;
            item.StartingRot = item.T.transform.rotation;
            item.Rb = item.T.GetComponent<Rigidbody>();
        }
        
        m_AgentGroup = new SimpleMultiAgentGroup();
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            m_AgentGroup.RegisterAgent(item.Agent);
        }
        ResetScene();
    }

    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            m_AgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }

        m_AgentGroup.AddGroupReward(-0.5f / MaxEnvironmentSteps);
    }

    public Vector3 GetRandomSpawnPos()
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;
        while (foundNewSpawnLocation == false)
        {
            var randomPosX = Random.Range(-areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.x * m_PushBlockSettings.spawnAreaMarginMultiplier);

            var randomPosZ = Random.Range(-areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.z * m_PushBlockSettings.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            if (Physics.CheckBox(randomSpawnPos, new Vector3(1.5f, 0.01f, 1.5f)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        return randomSpawnPos;
    }

    void ResetBlock(BlockInfo block)
    {
        block.T.position = GetRandomSpawnPos();
        block.Rb.linearVelocity = Vector3.zero;
        block.Rb.angularVelocity = Vector3.zero;
    }

    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); 
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void ScoredAGoal(Collider col, float score)
    {
        print($"Scored {score} on {gameObject.name}");
        m_NumberOfRemainingBlocks--;
        bool done = m_NumberOfRemainingBlocks == 0;
        col.gameObject.SetActive(false);
        m_AgentGroup.AddGroupReward(score);
        StartCoroutine(GoalScoredSwapGroundMaterial(m_PushBlockSettings.goalScoredMaterial, 0.5f));

        if (done)
        {
            m_AgentGroup.EndGroupEpisode();
            ResetScene();
        }
    }

    Quaternion GetRandomRot()
    {
        return Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
    }

    public void ResetScene()
    {
        m_ResetTimer = 0;
        
        // Clear old messages from the previous episode so they don't carry over!
        currentMessages.Clear();

        var rotation = Random.Range(0, 4);
        var rotationAngle = rotation * 90f;
        area.transform.Rotate(new Vector3(0f, rotationAngle, 0f));

        foreach (var item in AgentsList)
        {
            var pos = UseRandomAgentPosition ? GetRandomSpawnPos() : item.StartingPos;
            var rot = UseRandomAgentRotation ? GetRandomRot() : item.StartingRot;

            item.Agent.transform.SetPositionAndRotation(pos, rot);
            item.Rb.linearVelocity = Vector3.zero;
            item.Rb.angularVelocity = Vector3.zero;
        }

        foreach (var item in BlocksList)
        {
            var pos = UseRandomBlockPosition ? GetRandomSpawnPos() : item.StartingPos;
            var rot = UseRandomBlockRotation ? GetRandomRot() : item.StartingRot;

            item.T.transform.SetPositionAndRotation(pos, rot);
            item.Rb.linearVelocity = Vector3.zero;
            item.Rb.angularVelocity = Vector3.zero;
            item.T.gameObject.SetActive(true);
        }
        
        m_NumberOfRemainingBlocks = BlocksList.Count;
    }
}