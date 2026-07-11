using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;
using Lihaj.CommMAPOCA;

/// <summary>
/// Environment controller for the communication-enabled cooperative PushBlock
/// (Comm-MAPOCA benchmark variant). Mirrors the stock PushBlockEnvController
/// exactly (rewards, resets, timing), with two differences: agents are
/// CommPushAgentCollab, and the arena's CommChannel board is cleared on reset.
/// </summary>
public class CommPushBlockEnvController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public CommPushAgentCollab Agent;
        [HideInInspector] public Vector3 StartingPos;
        [HideInInspector] public Quaternion StartingRot;
        [HideInInspector] public Rigidbody Rb;
    }

    [System.Serializable]
    public class BlockInfo
    {
        public Transform T;
        [HideInInspector] public Vector3 StartingPos;
        [HideInInspector] public Quaternion StartingRot;
        [HideInInspector] public Rigidbody Rb;
    }

    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 25000;

    [HideInInspector] public Bounds areaBounds;

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

    [Header("Communication")]
    [Tooltip("The Comm-MAPOCA channel for this arena. Auto-found on this GameObject if left null.")]
    public CommChannel commChannel;

    [Header("Curriculum")]
    [Tooltip("The green goal zone. Early lessons spawn blocks NEAR it (short pushes ignite learning); later lessons spawn fully randomly. If null, near-goal spawning is disabled.")]
    public Transform goalZone;

    [Tooltip("Minimum flat distance from the goal zone's center that near-goal spawns keep, so a block never starts ON the strip (instant score, nothing learned). Should be ~strip half-width + block size + margin.")]
    public float minGoalDistance = 5f;

    // Per-lesson difficulty table, selected by the 'pb_lesson' environment parameter.
    // Blocks are picked BY TAG (blockSmall/blockLarge/blockVeryLarge), so the
    // Inspector order of BlocksList does not matter. spread: 0 = beside the goal,
    // 1 = fully random spawn.
    // v3: every rung KEEPS the mastered blocks and adds one new element.
    // (v2's L3 = one lone large removed all familiar objects; the blockLarge
    // sensor channel had never activated before, so agents wandered at -0.5
    // for 1.3M steps without ever engaging it.)
    //                                    L0    L1  L2  L3    L4  L5    L6  L7
    static readonly int[]   kSmalls  = {  1,    1,  2,  2,    2,  2,    2,  2 };
    static readonly int[]   kLarges  = {  0,    0,  0,  1,    1,  1,    2,  2 };
    static readonly int[]   kVLarges = {  0,    0,  0,  0,    0,  1,    1,  2 };
    static readonly float[] kSpread  = { 0.15f, 1f, 1f, 0.3f, 1f, 0.3f, 1f, 1f };

    private int m_NumberOfRemainingBlocks;
    private SimpleMultiAgentGroup m_AgentGroup;
    private int m_ResetTimer;

    void Start()
    {
        if (commChannel == null)
            commChannel = GetComponent<CommChannel>();

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

        // Hurry-up penalty (identical to stock)
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

    /// <summary>Wire the blocks' GoalDetectTrigger onTriggerEnterEvent to this.</summary>
    public void ScoredAGoal(Collider col, float score)
    {
        // Console spam once scoring becomes frequent; re-enable for debugging if needed.
        // print($"Scored {score} on {gameObject.name}");
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

        // Fresh episode = fresh message board
        if (commChannel != null)
            commChannel.ClearBoard();

        // Random platform rotation
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

        // --- CURRICULUM: which blocks participate, and where they spawn ---
        // 'pb_lesson' (set per lesson in the curriculum YAML) indexes the difficulty
        // table above. Default -1 = no curriculum = all blocks, fully random spawn,
        // so non-curriculum scenes behave exactly as before.
        int lesson = Mathf.RoundToInt(
            Academy.Instance.EnvironmentParameters.GetWithDefault("pb_lesson", -1f));

        int wantSmall, wantLarge, wantVLarge;
        float spread;
        if (lesson < 0)
        {
            wantSmall = wantLarge = wantVLarge = int.MaxValue; // everything
            spread = 1f;
        }
        else
        {
            lesson = Mathf.Clamp(lesson, 0, kSmalls.Length - 1);
            wantSmall = kSmalls[lesson];
            wantLarge = kLarges[lesson];
            wantVLarge = kVLarges[lesson];
            spread = kSpread[lesson];
        }

        int activeCount = 0;
        foreach (var item in BlocksList)
        {
            bool active;
            if (item.T.CompareTag("blockSmall"))
                active = wantSmall-- > 0;
            else if (item.T.CompareTag("blockLarge"))
                active = wantLarge-- > 0;
            else if (item.T.CompareTag("blockVeryLarge"))
                active = wantVLarge-- > 0;
            else
                active = true; // untagged/custom blocks always participate

            if (active)
            {
                var pos = UseRandomBlockPosition ? GetCurriculumSpawnPos(spread) : item.StartingPos;
                var rot = UseRandomBlockRotation ? GetRandomRot() : item.StartingRot;

                item.T.transform.SetPositionAndRotation(pos, rot);
                item.Rb.linearVelocity = Vector3.zero;
                item.Rb.angularVelocity = Vector3.zero;
                item.T.gameObject.SetActive(true);
                activeCount++;
            }
            else
            {
                // Not part of this lesson: park it inactive (also removes it from
                // sensors and from the remaining-blocks win condition).
                item.T.gameObject.SetActive(false);
            }
        }

        m_NumberOfRemainingBlocks = activeCount;
    }

    /// <summary>
    /// Block spawn position interpolated between "beside the goal zone" (spread 0)
    /// and "anywhere in the arena" (spread 1). Short pushes in early lessons give
    /// frequent accidental scores, igniting the reward bootstrap.
    /// </summary>
    Vector3 GetCurriculumSpawnPos(float spread)
    {
        var randomPos = GetRandomSpawnPos();
        if (goalZone == null || spread >= 0.999f)
            return randomPos;
        var p = Vector3.Lerp(goalZone.position, randomPos, Mathf.Clamp01(spread));

        // Never inside the goal zone: enforce a minimum flat distance so the
        // block always needs an actual push to score.
        var flat = p - goalZone.position;
        flat.y = 0f;
        if (flat.sqrMagnitude < minGoalDistance * minGoalDistance)
        {
            var dirFlat = randomPos - goalZone.position;
            dirFlat.y = 0f;
            var dir = dirFlat.sqrMagnitude > 0.01f ? dirFlat.normalized : transform.forward;
            p = goalZone.position + dir * minGoalDistance;
        }

        p.y = 1f;
        return p;
    }
}
