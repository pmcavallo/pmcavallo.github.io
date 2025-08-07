
# 🤖 Agentic AI Prototype – Natural Language Email & Calendar Assistant (Streamlit + ChatGPT + Google API)



This project demonstrates the implementation of an AI-augmented assistant built with Streamlit that connects to Gmail and Google Calendar via OAuth, interprets natural language commands, and executes intelligent actions.

The user can:
- Send emails
- Retrieve calendar events
- Use intuitive commands like "Show me my next 3 calendar events"

This system mimics the behavior of an **agentic AI**, combining user input parsing, secure API authentication, and real-time interaction with Google services.

**Tools:** Streamlit, Google Calendar API, Gmail API, OAuth 2.0, Python

---

## 🔐 Step 1: OAuth 2.0 Setup

We created a Google Cloud project and enabled the **Gmail API** and **Google Calendar API**.  
The generated `credentials.json` was placed in the working directory and included client secrets necessary for OAuth.

To enable OAuth securely:

```bash
# Delete the token if scopes change
rm token.json
```

This triggers a fresh consent screen allowing new scopes (permissions). Without this step, your app may silently reuse outdated tokens.

---

## 📁 Step 2: Project Structure

Here’s the directory:

```
agentic-ai/
├── streamlit_agentic_ai.py
├── credentials.json
├── token.json
├── .env
```

- `streamlit_agentic_ai.py`: Main app logic
- `.env`: Stores sensitive environment variables (e.g., OpenAI key)
- `credentials.json`: Google Cloud client credentials
- `token.json`: Generated upon OAuth consent

---

## 📥 Step 3: Google Authentication Flow

Once the `token.json` was deleted, the app prompted OAuth consent again. This screen ensures Agentic AI has permissions to:
- Read calendar events
- Send emails via Gmail

![OAuth Screen](attachment)

Once granted, the app created a new `token.json` — allowing authenticated access.

---

## 🧠 Step 4: Command Parsing

The Streamlit UI accepts **natural language instructions**, which are parsed into structured JSON-like intents.  
Initial limitations required JSON-formatted input. We extended parsing to support plain English, such as:

```text
Show me my next 3 calendar events
```

Upon submission, the backend:
1. Matches intent using regex patterns
2. Extracts arguments (e.g., number of events)
3. Calls the appropriate calendar function

Example success output:

```
Your next 3 calendar events are:
1. 2025-08-06 — Energy Bill Payment
2. 2025-08-11 — Chase CC Payment
3. 2025-08-13 — RTG Payment $121
```

---

## 🧪 Step 5: Debugging + Fixes

Several iterations were needed:
- ❌ SyntaxError from incomplete `try...except` block
- ❌ App ran without prompting auth (due to leftover tokens)
- ❌ `credentials.json` file not found
- ✅ All resolved with strategic file resets and logic repairs

---

## ✅ Current Functionality

You can now issue voice or text commands such as:

- **"Send an email to John about tomorrow’s meeting"**
- **"Check if I’m free on August 10th"**
- **"Cancel my 2 PM meeting"**
- **"Show me my next 3 calendar events"**

Each is parsed and routed to the correct function.

---

## 📌 Next Steps

- [ ] Extend natural language parser
- [ ] Add confirmation prompts before email/calendar actions
- [ ] Track historical interactions using session memory
- [ ] Add Slack or SMS integration for alerts

---

## 📸 Screenshots

Below are some images captured during this session (manually attach if uploading to GitHub):

- ✅ Calendar Output
- ⚠️ OAuth Consent Prompt
- ❌ Script Execution Errors
- ✅ Successful Launch

---

## 🧠 Final Thoughts

This project reflects a working **agentic architecture** — one where the AI can interpret, reason, and act across multiple tools in a secure environment.

Stay tuned for further enhancements in autonomy and cross-tool interactions!

