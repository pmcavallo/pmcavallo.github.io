---
layout: post
title: Agentic AI, Natural Language Email & Calendar Assistant (LangChain + Streamlit + ChatGPT + Google API)
date: 2025-07-12
---

This project demonstrates the implementation of an AI-augmented assistant built with Streamlit and powered by LangChain. It connects to Gmail and Google Calendar via OAuth, interprets natural language commands using an LLM agent, and executes intelligent actions such as sending emails or retrieving upcoming events.

The user can:
- Send emails
- Retrieve calendar events
- Use intuitive commands like "Show me my next 3 calendar events"

This system mimics the behavior of an **agentic AI**, combining user input parsing, secure API authentication, and real-time interaction with Google services.

**Tools:** LangChain, Streamlit, Google Calendar API, Gmail API, OAuth 2.0, Python

![ai](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/ai1.png?raw=true) 

---

## Step 1: OAuth 2.0 Setup

I created a Google Cloud project and enabled the **Gmail API** and **Google Calendar API**.  
The generated `credentials.json` was placed in the working directory and included client secrets necessary for OAuth.

To enable OAuth securely:

```bash
# Delete the token if scopes change
rm token.json
```

This triggers a fresh consent screen allowing new scopes (permissions). Without this step, the app may silently reuse outdated tokens.

```python
# Quick sanity check: is there an existing OAuth token?
print("token.json exists?", os.path.exists("token.json"))
```

---

## Step 2: Project Structure

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

## Step 3: Google Authentication Flow

Once the `token.json` was deleted, the app prompted OAuth consent again. This screen ensures Agentic AI has permissions to:
- Read calendar events
- Send emails via Gmail

Once granted, the app created a new `token.json` — allowing authenticated access.

---

## Step 4: Running the App

```text
streamlit run streamlit_agentic_ai.py
```

🔍 This command launches the Streamlit application defined in the file streamlit_agentic_ai.py.

Here’s what happens under the hood:
- Starts a local development server on localhost.
- Executes the Python script (streamlit_agentic_ai.py) and builds a web interface defined in it.
- Renders interactive UI components (text inputs, buttons, outputs) in the default browser.
- Listens for user input, passes it to the backend logic (intent recognition + tool execution), and displays the results on the same page.

Below is a portion of the streamlit_agentic_ai.py code:

```python

# -------------------------------
# Setup environment variables
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_TOKEN_PATH = "token.json"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",          # Send emails
    "https://www.googleapis.com/auth/calendar.readonly"    # Read calendar events
]

# -------------------------------
# Authenticate with Google
# -------------------------------
# -------------------------------
# Define LangChain Tool
# -------------------------------
# -------------------------------
# Agent Setup
# -------------------------------
# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🤖 Agentic AI Prototype")
user_input = st.text_area("📬 What would you like the agent to do?")

def parse_natural_language_input(text):
    try:
        to = re.search(r"email to (\S+)", text).group(1)
        subject = re.search(r"subject ['\"](.+?)['\"]", text).group(1)
        body = re.search(r"say: ['\"](.+?)['\"]", text).group(1)
        
        return {
            "to": to,
            "subject": subject,
            "body": body
        }
    except Exception as e:
        raise ValueError(f"Could not parse input. Please use the format:\n"
                         "Send an email to [recipient] with the subject '[subject]' and say: '[message]'\n\n"
                         f"Error: {e}")

if st.button("Submit"):
    try:
        # Try structured email first
        parsed_input = None
        try:
            parsed_input = json.loads(user_input)
        except:
            pass

        if parsed_input and all(k in parsed_input for k in ["to", "subject", "body"]):
            st.write(f"Sending email to: {parsed_input['to']}")
            st.write(f"Subject: {parsed_input['subject']}")
            st.write(f"Body: {parsed_input['body']}")
            
            send_email(
                to=parsed_input["to"],
                subject=parsed_input["subject"],
                body=parsed_input["body"]
            )
            st.success("Email sent successfully!")

        else:
            # Run full natural language request through the agent
            with st.spinner("Thinking..."):
                response = agent.run(user_input)
            st.write(response)

    except Exception as e:
        st.error(f"Failed to process input or send command:\n{e}")


```


---

## Step 5: Command Parsing

The Streamlit UI accepts **natural language instructions**, which are parsed into structured JSON-like intents.  
Initial limitations required JSON-formatted input. I extended parsing to support plain English, such as:

```text
Show me my next 3 calendar events
```
Upon submission, the backend:
1. Matches intent using regex patterns
2. Extracts arguments (e.g., number of events)
3. Calls the appropriate calendar function

OR

```text
Send an email to your_email@gmail.com with the subject "Follow-up from today’s session" and say: "Hi Paulo, just confirming everything’s working great!"
```

Upon submission, the backend:
1. Matches email-related intent using regex patterns (e.g., detects the “Send an email to...” phrase).
2. Extracts structured arguments such as:
    - to: recipient's email address
    - subject: email subject
    - body: message content

These are parsed from natural language and converted into a valid JSON payload.

3. Calls the email-sending function, which authenticates with Gmail API and sends the message via the user’s account.

##  LangChain-Powered Natural Language Intelligence

This Agentic AI prototype uses **LangChain** to translate the natural language instructions into action.

- If we provide a structured JSON payload like:
  ```json
  {"to": "friend@example.com", "subject": "Hello!", "body": "It was great chatting with you."}

The LangChain agent interprets your intent and chooses the appropriate tool:
- send_email – sends emails via Gmail (expects JSON with to, subject, body)
- list_calendar_events – retrieves the next five Google Calendar events

How it works (simplified code flow)

```python
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

# Prepare the tools:
email_tool = Tool("send_email", safe_send_email, "Send email with JSON payload")
calendar_tool = Tool("list_calendar_events", lambda _: list_upcoming_events(), "List upcoming calendar events")

# Initialize the LangChain agent:
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent([email_tool, calendar_tool], llm, agent="zero-shot-react-description")

# In the app:
if is_valid_email_json(user_input):
    send_email(**json.loads(user_input))
else:
    response = agent.run(user_input)
    st.write(response)

```

This setup lets me send emails or fetch my calendar just by typing what I want — no rigid form needed.

**Decision Note:** Opted for **LangChain with ChatOpenAI (GPT-4)** to orchestrate agents and tools because its structured workflow and prompt chaining provide robustness and easier scaling compared to manual regex handling or raw LLM prompts.


---
## ✅ Current Functionality

The functionality with the 2 currently present scopes are:

```pythom
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",          # Send emails
    "https://www.googleapis.com/auth/calendar.readonly"    # Read calendar events
]
```

📧 Send Emails via Gmail
- **Send an email on my behalf through the Gmail API**
- **Email content is parsed and formatted as a JSON payload**
- **The message is sent through the authenticated Gmail account**

📅 Retrieve Calendar Events from Google Calendar
- **Fetch upcoming calendar events from your primary Google calendar**
- **Results include event date, title, and are listed in chronological order**

## ✅ Upcoming Functionality

Next goals are to issue voice or text commands such as:

- **"Send an email to John about tomorrow’s meeting"**
- **"Check if I’m free on August 10th"**
- **"Cancel my 2 PM meeting"**
- **"Show me my next 3 calendar events"**

Each will be parsed and routed to the correct function.

**Considered Alternatives:**  
- Rule-based or regex-only parsing—too fragile and hard to expand.  
- Direct OpenAI API calls without LangChain—lacked tool abstraction and manageability.

---

## 📌 Next Steps

- [ ] Extend natural language parser
- [ ] Add confirmation prompts before email/calendar actions
- [ ] Track historical interactions using session memory
- [ ] Add Slack or SMS integration for alerts

---


## Final Thoughts

This project reflects a working **agentic architecture** — one where the AI can interpret, reason, and act across multiple tools in a secure environment.

Stay tuned for further enhancements in autonomy and cross-tool interactions!

