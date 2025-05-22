# import os, sys, ...
import os
# import utils module (numpy, random, math)
from dotenv import load_dotenv
import hmac, hashlib

# import model module (torch, transformers, etc.)
import httpx
from fastapi import FastAPI, HTTPException, Request, Query

# import local library

# import local file .py
from infer import inference

load_dotenv(override=True)

FB_APP_SECRET = os.environ.get("FB_APP_SECRET", "")
FB_ACCESS_TOKEN = os.environ.get("FB_ACCESS_TOKEN", "")
FB_VERIFY_TOKEN = os.environ.get("FB_VERIFY_TOKEN", "")
if not FB_APP_SECRET or not FB_ACCESS_TOKEN or not FB_VERIFY_TOKEN:
    raise ValueError("Missing environment variables. Please set FB_APP_SECRET, FB_ACCESS_TOKEN, and FB_VERIFY_TOKEN.")

# Init the app
app = FastAPI()

# Middleware to verify Facebook signature
@app.middleware("http")
async def facebook_signature_middleware(request: Request, call_next):
    if request.url.path == "/webhook/" and request.method == "POST":
        signature = request.headers.get("x-hub-signature")

        if not signature:
            raise HTTPException(status_code=403, detail="Missing X-Hub-Signature")

        body = await request.body()

        try:
            method, sign_hash = signature.split("=")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid signature format")

        real_hash = hmac.new(
            key=FB_APP_SECRET.encode("utf-8"),
            msg=body,
            digestmod=hashlib.sha1
        ).hexdigest()

        if not hmac.compare_digest(sign_hash, real_hash):
            raise HTTPException(status_code=403, detail="Invalid signature")

        # Re-inject body into the request stream
        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive  # Hacky, but necessary to reuse body

    response = await call_next(request)
    return response

@app.get("/")
async def read_root():
    return {"Hello": "World"}


# GET handler for Facebook webhook verification
@app.get("/webhook/")
def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == FB_VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return int(hub_challenge)
    else:
        print("Webhook verification failed.")
        raise HTTPException(status_code=403, detail="Verification failed")

# POST handler to receive messages
@app.post("/webhook/")
# @app.post("/webhook")
# @app.post("/webhook/")
async def webhook(request: Request):
    print("Webhook messaging step.")
    chat_data = await request.json()

    if chat_data.get("object") == "page":
        for page_body in chat_data.get("entry", []):
            for message_obj in page_body.get("messaging", []):
                if "message" in message_obj:
                    if message_obj["message"].get("is_echo"):
                        continue
                    try:
                        message = get_message(message_obj)
                        response = inference(query=message).content

                        if isinstance(response, str):
                            await send_message(
                                recipient_id=message_obj["sender"]["id"],
                                message=response
                            )
                        else:
                            await send_message(
                                recipient_id=message_obj["sender"]["id"],
                                message="Sorry, I couldn't process your request."
                            )
                    except Exception as e:
                        print(f"Error while processing message: {e}")
                        await send_message(
                            recipient_id=message_obj["sender"]["id"],
                            message="Oops! Something went wrong."
                        )
        return {"status": "ok"}  # ✅ ensure this is always returned
    else:
        raise HTTPException(status_code=400, detail="Not a page subscription")

# Get message content
def get_message(message_obj):
    message = message_obj.get("message", {}).get("text", "")
    return message

# Send a reply message
async def send_message(recipient_id: str, message: str):
    MAX_MSG_LENGTH = 2000
    message_chunks = [message[i:i + MAX_MSG_LENGTH] for i in range(0, len(message), MAX_MSG_LENGTH)]

    async with httpx.AsyncClient() as client:
        for chunk in message_chunks:
            message_data = {
                "recipient": {"id": recipient_id},
                "message": {"text": chunk}
            }
            response = await client.post(
                url="https://graph.facebook.com/v3.2/me/messages",
                params={"access_token": FB_ACCESS_TOKEN},
                json=message_data
            )
            if response.status_code == 200:
                print("Message sent successfully.")
            else:
                print(f"Message failed - {response.status_code}: {response.text}")
                break
