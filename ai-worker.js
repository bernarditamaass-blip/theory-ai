/* THEORY // AI — modelo local. No utiliza claves API ni un servidor de inferencia. */
let generator = null;
let loading = false;
let chosenDevice = 'sin iniciar';

async function loadModel() {
  if (generator) return;
  if (loading) {
    while (loading) await new Promise(r => setTimeout(r, 250));
    if (generator) return;
  }
  loading = true;
  try {
    self.postMessage({ type: 'status', status: 'Descargando el motor de IA…' });
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2');
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
    env.useFS = false;

    const model = 'onnx-community/Qwen2.5-0.5B-Instruct';
    const attempts = [];
    if (self.navigator && self.navigator.gpu) {
      attempts.push({ device: 'webgpu', dtype: 'q4f16' });
      attempts.push({ device: 'webgpu', dtype: 'q4' });
    }
    attempts.push({ device: 'wasm', dtype: 'q4' });
    let lastError;
    for (const option of attempts) {
      try {
        self.postMessage({ type: 'status', status: option.device === 'webgpu' ? 'Preparando IA con aceleración gráfica…' : 'Preparando IA compatible (CPU)…' });
        generator = await pipeline('text-generation', model, {
          device: option.device,
          dtype: option.dtype,
          progress_callback: p => {
            if (p && p.status) self.postMessage({ type: 'status', status: p.status === 'progress' ? `Descargando modelo: ${Math.round(p.progress || 0)}%` : `Modelo: ${p.status}` });
          }
        });
        chosenDevice = option.device;
        break;
      } catch (e) {
        lastError = e;
        generator = null;
      }
    }
    if (!generator) throw lastError || new Error('No se pudo cargar el modelo en este dispositivo.');
    self.postMessage({ type: 'ready', device: chosenDevice, status: `IA lista (${chosenDevice === 'webgpu' ? 'gráficos' : 'CPU'}).` });
  } finally {
    loading = false;
  }
}

self.onmessage = async event => {
  const data = event.data || {};
  if (data.type === 'init') {
    try { await loadModel(); }
    catch (e) { self.postMessage({ type: 'error', error: friendlyError(e) }); }
    return;
  }
  if (data.type === 'generate') {
    try {
      await loadModel();
      const messages = [
        { role: 'system', content: data.system || 'Eres THEORY, una persona virtual con personalidad propia. Responde en español, breve, natural y sin repetir frases. Tu ánimo modifica sutilmente el tono, pero no inventes datos.' },
        ...(Array.isArray(data.history) ? data.history.slice(-8) : []),
        { role: 'user', content: data.message || '' }
      ];
      self.postMessage({ type: 'status', status: 'THEORY está pensando…' });
      const output = await generator(messages, {
        max_new_tokens: 150,
        temperature: 0.78,
        do_sample: true,
        repetition_penalty: 1.12
      });
      let result = output && output[0] && output[0].generated_text;
      let reply = '';
      if (Array.isArray(result)) {
        const last = result[result.length - 1];
        reply = typeof last === 'string' ? last : (last && last.content) || '';
      } else if (typeof result === 'string') {
        reply = result;
      }
      reply = String(reply).trim();
      if (!reply) throw new Error('El modelo no devolvió una respuesta.');
      self.postMessage({ type: 'reply', reply: reply.slice(-1800), device: chosenDevice });
    } catch (e) {
      self.postMessage({ type: 'error', error: friendlyError(e) });
    }
  }
};

function friendlyError(error) {
  const message = String(error && error.message || error || 'Error desconocido');
  if (/fetch|network|load|404/i.test(message)) return 'No se pudo descargar el modelo. Comprueba la conexión e inténtalo otra vez.';
  if (/memory|allocation|gpu|device|out of/i.test(message)) return 'Este dispositivo no tiene memoria o aceleración suficiente para cargar el modelo.';
  return `No se pudo iniciar la IA local: ${message.slice(0, 220)}`;
}
