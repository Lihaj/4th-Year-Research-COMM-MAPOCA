using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Lihaj.CommMAPOCA
{
    /// <summary>
    /// Base class for agents that communicate through a <see cref="CommChannel"/>.
    ///
    /// What it does automatically:
    ///  - configures the required BufferSensorComponent as the "CommBuffer" the
    ///    comm_mapoca trainer looks for (row size = messageSize + 1, capacity from the channel);
    ///  - appends this agent's incoming messages to that sensor every observation step;
    ///  - broadcasts the first messageSize continuous actions, tanh-bounded to [-1, 1],
    ///    to the channel every action step;
    ///  - clears this agent's posted message when its episode begins.
    ///
    /// Derive from it and implement the *Comm* virtual methods instead of the usual
    /// Agent overrides (those are sealed here so the comm plumbing always runs):
    ///  - <see cref="CollectAgentObservations"/> instead of CollectObservations
    ///  - <see cref="OnCommActionReceived"/>     instead of OnActionReceived
    ///  - <see cref="OnCommInitialize"/>         instead of Initialize
    ///  - <see cref="OnCommEpisodeBegin"/>       instead of OnEpisodeBegin
    ///
    /// To send a scripted message instead of the policy's learned one (e.g., a
    /// hand-coded ground-truth message while validating an environment), override
    /// <see cref="OverrideOutgoingMessage"/>, fill the buffer, and return true.
    /// </summary>
    [RequireComponent(typeof(BufferSensorComponent))]
    public abstract class CommAgent : Agent
    {
        [Header("Comm-MAPOCA")]
        [Tooltip("The message board this agent talks through. Leave null to disable communication for this agent.")]
        public CommChannel channel;

        [Tooltip("During Python-driven training, the comm_mapoca trainer can smuggle the receiver's attention weights into spare continuous action slots (after the message slots). Enable to read them and report to the channel for gizmo coloring. Weights are NOT meaningful in standalone ONNX inference.")]
        public bool readSmuggledAttention = false;

        /// <summary>Sensor name the Python trainer detects communication buffers by.</summary>
        public const string CommSensorName = "CommBuffer";

        BufferSensorComponent m_CommSensor;
        float[] m_OutgoingMessage;

        /// <summary>The last message this agent broadcast (read-only, for debugging/analysis).</summary>
        public System.ReadOnlySpan<float> LastSentMessage => m_OutgoingMessage;

        public sealed override void Initialize()
        {
            m_CommSensor = GetComponent<BufferSensorComponent>();
            if (channel != null)
            {
                // Initialize() runs before the Agent creates its sensors (Agent.cs order),
                // so configuring the BufferSensorComponent here takes effect.
                m_CommSensor.SensorName = CommSensorName;
                m_CommSensor.ObservableSize = channel.RowSize;
                m_CommSensor.MaxNumObservables = channel.maxMessagesPerReceiver;
                m_OutgoingMessage = new float[channel.messageSize];
                channel.RegisterAgent(this);
            }
            OnCommInitialize();
        }

        /// <summary>Your usual Initialize() logic goes here.</summary>
        protected virtual void OnCommInitialize() { }

        public sealed override void CollectObservations(VectorSensor sensor)
        {
            CollectAgentObservations(sensor);
            if (channel != null && m_CommSensor != null)
            {
                foreach (var row in channel.GetIncomingRows(this))
                    m_CommSensor.AppendObservation(row);
            }
        }

        /// <summary>Your usual CollectObservations() logic goes here (vector observations only — incoming messages are appended automatically).</summary>
        protected virtual void CollectAgentObservations(VectorSensor sensor) { }

        public sealed override void OnActionReceived(ActionBuffers actions)
        {
            if (channel != null && m_OutgoingMessage != null)
            {
                if (!OverrideOutgoingMessage(m_OutgoingMessage))
                {
                    // Learned message: first messageSize continuous actions, tanh-bounded so
                    // unbounded policy outputs cannot grow without limit and destabilize the
                    // (unnormalized) critic that also consumes these messages.
                    var cont = actions.ContinuousActions;
                    int n = Mathf.Min(m_OutgoingMessage.Length, cont.Length);
                    for (int i = 0; i < n; i++)
                        m_OutgoingMessage[i] = (float)System.Math.Tanh(cont[i]);
                }
                channel.PostMessage(this, m_OutgoingMessage);

                if (readSmuggledAttention)
                    ReportSmuggledAttention(actions);
            }
            OnCommActionReceived(actions);
        }

        /// <summary>Your usual OnActionReceived() logic goes here (movement etc.). The message slots have already been broadcast.</summary>
        protected virtual void OnCommActionReceived(ActionBuffers actions) { }

        /// <summary>
        /// Fill <paramref name="message"/> (length = channel.messageSize) with a scripted
        /// message and return true to broadcast it INSTEAD of the policy's learned message.
        /// Default: return false (use the learned message). Intended for environment-side
        /// validation tools like ground-truth/hand-coded messages — keep such logic in your
        /// environment code, not in this package.
        /// </summary>
        protected virtual bool OverrideOutgoingMessage(float[] message) => false;

        public sealed override void OnEpisodeBegin()
        {
            channel?.ClearFor(this);
            OnCommEpisodeBegin();
        }

        /// <summary>Your usual OnEpisodeBegin() logic goes here.</summary>
        protected virtual void OnCommEpisodeBegin() { }

        /// <summary>
        /// Reads attention weights the trainer wrote into the continuous action slots after
        /// the message ([messageSize .. messageSize+k)) and reports them to the channel for
        /// gizmo coloring. Weight k corresponds to the k-th allowed sender (row order).
        /// </summary>
        void ReportSmuggledAttention(ActionBuffers actions)
        {
            var cont = actions.ContinuousActions;
            int spare = cont.Length - channel.messageSize;
            if (spare <= 0)
                return;
            var senders = channel.GetAllowedSenders(this);
            int n = Mathf.Min(spare, senders.Count);
            for (int k = 0; k < n; k++)
            {
                float w = Mathf.Clamp01(cont[channel.messageSize + k]);
                channel.ReportAttention(this, senders[k], w);
            }
        }

        protected virtual void OnDestroy()
        {
            if (channel != null)
                channel.UnregisterAgent(this);
        }
    }
}
