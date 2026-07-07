using System.Collections.Generic;
using UnityEngine;

namespace Lihaj.CommMAPOCA
{
    /// <summary>Which agents may talk to which.</summary>
    public enum CommTopologyMode
    {
        /// <summary>Every registered agent's message is delivered to every other registered agent.</summary>
        AllToAll,
        /// <summary>Only the explicit sender -> receiver pairs in <see cref="CommChannel.directedLinks"/> deliver.</summary>
        Directed
    }

    /// <summary>One explicit sender -> receiver edge for the Directed topology.</summary>
    [System.Serializable]
    public class DirectedLink
    {
        public CommAgent sender;
        public CommAgent receiver;
    }

    /// <summary>
    /// The central message board for one team / environment instance.
    ///
    /// Senders post a fixed-size float message every action step (see <see cref="CommAgent"/>);
    /// receivers read the board through their CommBuffer sensor. Each delivered row is
    /// [messageSize floats | presence flag]: presence 1 marks a real message, 0 marks a
    /// registered-but-silent sender slot, so the trainer's attention can mask empty slots
    /// explicitly instead of guessing from message magnitude.
    ///
    /// Row order is stable (sender registration order) so the trainer's attention weights
    /// keep a consistent meaning across steps.
    /// </summary>
    public class CommChannel : MonoBehaviour
    {
        [Header("Message Format")]
        [Tooltip("Number of floats in one message, EXCLUDING the presence flag. The CommBuffer sensor row size becomes messageSize + 1. Must match the trainer's message size.")]
        public int messageSize = 4;

        [Tooltip("Capacity (max rows) of each receiver's CommBuffer sensor. Must be >= the number of senders that can talk to one receiver.")]
        public int maxMessagesPerReceiver = 10;

        [Header("Topology")]
        public CommTopologyMode topology = CommTopologyMode.AllToAll;

        [Tooltip("Used only when topology = Directed: explicit sender -> receiver pairs.")]
        public List<DirectedLink> directedLinks = new List<DirectedLink>();

        [Tooltip("If false (recommended), an agent never receives its own message. Excluding self-messages keeps an agent's OWN sent message cleanly out of its observations, which the MA-POCA counterfactual baseline needs in order to marginalize it.")]
        public bool allowSelfMessages = false;

        [Header("Visualization")]
        [Tooltip("Draw communication flow lines in the Scene view (receiver -> sender, colored by attention weight when the trainer smuggles weights, otherwise flat).")]
        public bool showFlowGizmos = true;

        [Tooltip("Hide gizmo lines whose attention weight is below this threshold (reduces clutter). Lines without attention data are always drawn.")]
        [Range(0f, 1f)]
        public float gizmoAttentionCutoff = 0.05f;

        /// <summary>Registered agents in registration order — this order defines row order.</summary>
        readonly List<CommAgent> m_Agents = new List<CommAgent>();

        /// <summary>Latest message posted by each sender this episode.</summary>
        readonly Dictionary<CommAgent, float[]> m_Board = new Dictionary<CommAgent, float[]>();

        /// <summary>Attention weight per (receiver, sender), reported by receivers for gizmos.</summary>
        readonly Dictionary<(CommAgent receiver, CommAgent sender), float> m_Attention =
            new Dictionary<(CommAgent, CommAgent), float>();

        /// <summary>Size of one delivered row: message content + presence flag.</summary>
        public int RowSize => messageSize + 1;

        /// <summary>All registered agents (read-only, registration order).</summary>
        public IReadOnlyList<CommAgent> RegisteredAgents => m_Agents;

        /// <summary>Called by CommAgent.Initialize. Safe to call more than once.</summary>
        public void RegisterAgent(CommAgent agent)
        {
            if (agent != null && !m_Agents.Contains(agent))
                m_Agents.Add(agent);
        }

        public void UnregisterAgent(CommAgent agent)
        {
            m_Agents.Remove(agent);
            m_Board.Remove(agent);
        }

        /// <summary>Post (or overwrite) the sender's message for this step. The message is copied.</summary>
        public void PostMessage(CommAgent sender, float[] message)
        {
            if (sender == null || message == null)
                return;
            if (!m_Board.TryGetValue(sender, out var stored) || stored.Length != messageSize)
            {
                stored = new float[messageSize];
                m_Board[sender] = stored;
            }
            var n = Mathf.Min(messageSize, message.Length);
            System.Array.Copy(message, stored, n);
        }

        /// <summary>True if the topology delivers sender's messages to receiver.</summary>
        public bool CanCommunicate(CommAgent sender, CommAgent receiver)
        {
            if (sender == null || receiver == null)
                return false;
            if (!allowSelfMessages && ReferenceEquals(sender, receiver))
                return false;
            if (topology == CommTopologyMode.AllToAll)
                return true;
            foreach (var link in directedLinks)
            {
                if (ReferenceEquals(link.sender, sender) && ReferenceEquals(link.receiver, receiver))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// The senders whose slot appears in <paramref name="receiver"/>'s CommBuffer,
        /// in stable row order. Attention weight k (from the trainer) refers to element k here.
        /// </summary>
        public List<CommAgent> GetAllowedSenders(CommAgent receiver)
        {
            var senders = new List<CommAgent>();
            foreach (var candidate in m_Agents)
            {
                if (candidate == null || !candidate.gameObject.activeInHierarchy)
                    continue;
                if (CanCommunicate(candidate, receiver))
                    senders.Add(candidate);
            }
            return senders;
        }

        /// <summary>
        /// Build the rows to append to <paramref name="receiver"/>'s CommBuffer this step:
        /// one row per allowed sender, [content... | presence].
        /// </summary>
        public List<float[]> GetIncomingRows(CommAgent receiver)
        {
            var rows = new List<float[]>();
            foreach (var sender in GetAllowedSenders(receiver))
            {
                var row = new float[RowSize];
                if (m_Board.TryGetValue(sender, out var msg))
                {
                    System.Array.Copy(msg, row, messageSize);
                    row[messageSize] = 1f; // presence: sender has spoken this episode
                }
                // else: all zeros incl. presence flag = registered but silent slot
                rows.Add(row);
            }
            return rows;
        }

        /// <summary>Forget everything posted (e.g., on full environment reset).</summary>
        public void ClearBoard()
        {
            m_Board.Clear();
            m_Attention.Clear();
        }

        /// <summary>Forget one agent's posted message (called when that agent's episode begins).</summary>
        public void ClearFor(CommAgent agent)
        {
            m_Board.Remove(agent);
        }

        /// <summary>Receivers report trainer-smuggled attention weights here for gizmo coloring.</summary>
        public void ReportAttention(CommAgent receiver, CommAgent sender, float weight)
        {
            m_Attention[(receiver, sender)] = Mathf.Clamp01(weight);
        }

        void OnDrawGizmos()
        {
            if (!showFlowGizmos || !Application.isPlaying)
                return;

            foreach (var receiver in m_Agents)
            {
                if (receiver == null || !receiver.gameObject.activeInHierarchy)
                    continue;
                foreach (var sender in GetAllowedSenders(receiver))
                {
                    // only draw flows that carry a real message
                    if (!m_Board.ContainsKey(sender))
                        continue;

                    if (m_Attention.TryGetValue((receiver, sender), out var w))
                    {
                        if (w < gizmoAttentionCutoff)
                            continue;
                        var c = Color.Lerp(Color.red, Color.green, w);
                        c.a = 1f;
                        Gizmos.color = c;
                        Gizmos.DrawLine(receiver.transform.position, sender.transform.position);
                        Gizmos.DrawSphere(sender.transform.position, 0.5f * w);
                    }
                    else
                    {
                        Gizmos.color = Color.cyan;
                        Gizmos.DrawLine(receiver.transform.position, sender.transform.position);
                    }
                }
            }
        }
    }
}
