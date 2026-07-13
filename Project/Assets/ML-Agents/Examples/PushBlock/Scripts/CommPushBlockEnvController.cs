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
        [HideInInspector] public float StartingMass;
        [HideInInspector] public float PrevGoalDist;
        [HideInInspector] public Vector3 EpisodeStartPos;
        [HideInInspector] public float MaxDistFromStart;
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

    [Tooltip("Potential-based shaping for HEAVY blocks (blockLarge/blockVeryLarge): each physics step the team earns (previous - current block distance to goal) x this scale. Random exploration rarely completes a heavy push, so this dense signal ignites the skill; pushing goal-ward pays, pushing away costs. 0 disables (default: benchmark scenes must opt in explicitly). Requires goalZone.")]
    public float heavyShapingScale = 0f;

    [Tooltip("Video-exact shaping (adabeat.com PushBlock experiment): the team earns this reward EACH TIME a heavy block moves further from its episode-start position than ever before (a ratchet - only new record distances pay, so jiggling in place earns nothing). Use 0.001 with goal walls around the arena; 0 disables. Independent of goalZone.")]
    public float awayFromStartReward = 0f;

    // Per-lesson difficulty table, selected by the 'pb_lesson' environment parameter.
    // Blocks are picked BY TAG (blockSmall/blockLarge/blockVeryLarge), so the
    // Inspector order of BlocksList does not matter. spread: 0 = beside the goal,
    // 1 = fully random spawn.
    // v3: every rung KEEPS the mastered blocks and adds one new element.
    // (v2's L3 = one lone large removed all familiar objects; the blockLarge
    // sensor channel had never activated before, so agents wandered at -0.5
    // for 1.3M steps without ever engaging it.)
    // v4: L3/L6 are "light mass" rungs -- the newly-introduced heavy block
    // temporarily masses the same as a small block, so agents (who already know
    // how to push smalls) transfer that skill onto it immediately instead of
    // needing a rare, precisely-timed multi-agent push to ever move it at all.
    // The very next lesson restores real mass with the seek-and-push habit
    // already ingrained. (Real L3/L5 mass requirements were previously stalling
    // for millions of steps: 2-agent co-location on the SAME heavy block, at the
    // SAME time, almost never happened by chance from a standing start.)
    // v5: L8 is a new "vlarge anywhere" rung (mirrors L5's role for large: same
    // blocks as L7, just spread=1 instead of near-goal). Without it, vlarge only
    // ever saw near-goal spawns (L6/L7) before MixedFive's full-random spawn --
    // agents pushed it fine when it landed near goal, but never learned to push
    // it from the far half of the arena at all, since that distance was never
    // practiced on its own before a 2nd large block also got added at the same time.
    //                                        L0    L1  L2  L3    L4    L5  L6    L7    L8  L9  L10
    static readonly int[]   kSmalls        = { 1,    1,  2,  2,    2,    2,  2,    2,    2,  2,  2 };
    static readonly int[]   kLarges        = { 0,    0,  0,  1,    1,    1,  1,    1,    1,  2,  2 };
    static readonly int[]   kVLarges       = { 0,    0,  0,  0,    0,    0,  1,    1,    1,  1,  2 };
    static readonly float[] kSpread        = { 0.15f, 1f, 1f, 0.3f, 0.3f, 1f, 0.3f, 0.3f, 1f, 1f, 1f };
    static readonly bool[]  kLightenLarge  = { false, false, false, true,  false, false, false, false, false, false, false };
    static readonly bool[]  kLightenVLarge = { false, false, false, false, false, false, true,  false, false, false, false };

    private float m_SmallBlockMass;
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
            item.StartingMass = item.Rb.mass;
            if (item.T.CompareTag("blockSmall"))
                m_SmallBlockMass = item.StartingMass;
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

        // Heavy-block "getting warmer" shaping (potential-based, like the Blind
        // Walker's): reward the team for heavy blocks getting CLOSER to the goal.
        if (heavyShapingScale != 0f && goalZone != null)
        {
            foreach (var item in BlocksList)
            {
                if (item.T == null || !item.T.gameObject.activeInHierarchy || !IsHeavy(item.T))
                    continue;
                float curr = FlatDistToGoal(item.T.position);
                m_AgentGroup.AddGroupReward((item.PrevGoalDist - curr) * heavyShapingScale);
                item.PrevGoalDist = curr;
            }
        }

        // Video-exact heavy-block shaping: +awayFromStartReward each time a heavy
        // block sets a NEW record distance from its episode-start position.
        if (awayFromStartReward != 0f)
        {
            foreach (var item in BlocksList)
            {
                if (item.T == null || !item.T.gameObject.activeInHierarchy || !IsHeavy(item.T))
                    continue;
                var flat = item.T.position - item.EpisodeStartPos;
                flat.y = 0f;
                float d = flat.magnitude;
                if (d > item.MaxDistFromStart + 0.01f)
                {
                    m_AgentGroup.AddGroupReward(awayFromStartReward);
                    item.MaxDistFromStart = d;
                }
            }
        }
    }

    static bool IsHeavy(Transform t)
    {
        return t.CompareTag("blockLarge") || t.CompareTag("blockVeryLarge");
    }

    float FlatDistToGoal(Vector3 pos)
    {
        var d = pos - goalZone.position;
        d.y = 0f;
        return d.magnitude;
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
        bool lightenLarge, lightenVLarge;
        if (lesson < 0)
        {
            wantSmall = wantLarge = wantVLarge = int.MaxValue; // everything
            spread = 1f;
            lightenLarge = lightenVLarge = false;
        }
        else
        {
            lesson = Mathf.Clamp(lesson, 0, kSmalls.Length - 1);
            wantSmall = kSmalls[lesson];
            wantLarge = kLarges[lesson];
            wantVLarge = kVLarges[lesson];
            spread = kSpread[lesson];
            lightenLarge = kLightenLarge[lesson];
            lightenVLarge = kLightenVLarge[lesson];
        }

        int activeCount = 0;

        // Heavy blocks get first claim on the near-goal spawn region: they're
        // what a new lesson is trying to bootstrap, while already-mastered small
        // blocks don't need to be tight to the goal. Placing them first means the
        // overlap-retry in GetSafeBlockSpawnPos pushes SMALLS out of the crowded
        // spot instead of the heavy block that actually needs proximity.
        var placementOrder = new List<BlockInfo>(BlocksList);
        placementOrder.Sort((a, b) => (IsHeavy(a.T) ? 0 : 1).CompareTo(IsHeavy(b.T) ? 0 : 1));

        foreach (var item in placementOrder)
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
                var pos = UseRandomBlockPosition ? GetSafeBlockSpawnPos(spread) : item.StartingPos;
                var rot = UseRandomBlockRotation ? GetRandomRot() : item.StartingRot;

                item.T.transform.SetPositionAndRotation(pos, rot);
                item.Rb.linearVelocity = Vector3.zero;
                item.Rb.angularVelocity = Vector3.zero;

                // Light-mass rungs (see kLightenLarge/kLightenVLarge): the block
                // still carries its real tag/sensor signature, only the physics
                // response changes. Always set explicitly (not just when
                // lightened) so a real-mass lesson can't inherit a stale light
                // mass left over from an earlier reset.
                if (item.T.CompareTag("blockLarge"))
                    item.Rb.mass = lightenLarge ? m_SmallBlockMass : item.StartingMass;
                else if (item.T.CompareTag("blockVeryLarge"))
                    item.Rb.mass = lightenVLarge ? m_SmallBlockMass : item.StartingMass;

                item.T.gameObject.SetActive(true);
                activeCount++;

                // Seed the shaping baselines so the first deltas after reset are ~0
                if (goalZone != null)
                    item.PrevGoalDist = FlatDistToGoal(item.T.position);
                item.EpisodeStartPos = item.T.position;
                item.MaxDistFromStart = 0f;
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
    /// Block spawn position guaranteed not to overlap any goal-tagged trigger.
    /// With goal strips along the walls, a block spawning ON a strip would score
    /// instantly at reset and teach nothing. (Agents may still spawn on strips
    /// harmlessly -- scoring triggers live on the blocks.)
    /// </summary>
    Vector3 GetSafeBlockSpawnPos(float spread)
    {
        var pos = GetCurriculumSpawnPos(spread);
        for (int i = 0; i < 25 && BlockPosTouchesGoal(pos); i++)
            pos = GetCurriculumSpawnPos(spread);
        return pos;
    }

    static readonly Collider[] s_OverlapBuf = new Collider[16];

    bool BlockPosTouchesGoal(Vector3 pos)
    {
        // Generous probe (covers even the very-large blocks' footprint).
        int n = Physics.OverlapBoxNonAlloc(pos, new Vector3(2f, 1.5f, 2f), s_OverlapBuf);
        for (int i = 0; i < n; i++)
        {
            if (s_OverlapBuf[i].CompareTag("goal"))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Block spawn position in a tight distance band from the goal zone (spread 0)
    /// widening toward "anywhere in the arena" (spread 1, which bypasses this function
    /// entirely). Short pushes in early lessons give frequent accidental scores,
    /// igniting the reward bootstrap.
    /// </summary>
    Vector3 GetCurriculumSpawnPos(float spread)
    {
        if (goalZone == null || spread >= 0.999f)
            return GetRandomSpawnPos();

        // Distance band, NOT a lerp toward an arbitrary random point: lerping only
        // enforced a minimum distance from goal, never a maximum, so whenever the
        // random point landed on the far side of the (~25-unit) arena, even a tight
        // spread like 0.15 could still leave the block several units out. Sampling
        // directly within [minGoalDistance, minGoalDistance + spread*10] keeps every
        // near-goal lesson consistently tight regardless of arena rotation.
        float maxDist = minGoalDistance + spread * 10f;

        for (int i = 0; i < 25; i++)
        {
            // Direction from a legally-spawnable random point, so we stay inside the
            // arena and away from walls; only the distance is overridden.
            var randomPos = GetRandomSpawnPos();
            var dirFlat = randomPos - goalZone.position;
            dirFlat.y = 0f;
            var dir = dirFlat.sqrMagnitude > 0.01f ? dirFlat.normalized : transform.forward;
            var p = goalZone.position + dir * Random.Range(minGoalDistance, maxDist);

            p.y = 1f;

            // p is a fresh point near goalZone, not randomPos itself -- randomPos was
            // collision-checked but only used for direction, so p still needs its own
            // check (this is the fix for blocks spawning stacked at low spread).
            if (!Physics.CheckBox(p, new Vector3(1.5f, 0.01f, 1.5f)))
                return p;
        }

        // Repeatedly collided near the goal anchor: fall back to a plain
        // collision-checked random spot rather than spawning on top of something.
        return GetRandomSpawnPos();
    }
}
