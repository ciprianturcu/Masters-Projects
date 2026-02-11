// app.js
const API_BASE = 'https://ch4p1uezu1.execute-api.us-east-1.amazonaws.com/prod';

/**
 * POST /submit-journal – Create a new journal entry.
 * If 'file' is a File instance, convert it to base64; otherwise, send just { text }.
 */
async function createEntry(text, file) {
  // Base payload always contains at least { text }
  const body = { text };

  // Only try FileReader if 'file' is a genuine File
  if (file && file instanceof File) {
    const reader = new FileReader();
    return new Promise((resolve, reject) => {
      reader.onload = async () => {
        // reader.result looks like "data:<mime>;base64,<base64data>"
        const dataURL = reader.result;
        const base64Data = dataURL.split(',')[1];
        body.file_data = base64Data;
        body.file_name = file.name;
        body.content_type = file.type;

        try {
          const res = await fetch(`${API_BASE}/submit-journal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
          if (!res.ok) {
            const errText = await res.text();
            reject(`Error ${res.status}: ${errText}`);
            return;
          }
          const json = await res.json();
          resolve(json.entry_id);
        } catch (e) {
          reject(e);
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file); 
    });
  }

  // If no valid File, just send text
  const res = await fetch(`${API_BASE}/submit-journal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Error ${res.status}: ${errText}`);
  }
  const json = await res.json();
  return json.entry_id;
}


/**
 * GET /list-journals – Fetch list of all entries.
 */
async function fetchEntries() {
  const res = await fetch(`${API_BASE}/list-journals`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return await res.json();
}

/**
 * GET /journal/{id} – Fetch a single entry’s details.
 */
async function fetchEntryById(entryId) {
  const res = await fetch(`${API_BASE}/journal/${entryId}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return await res.json();
}

/** Utility to read URL query parameter “id”. */
function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}
