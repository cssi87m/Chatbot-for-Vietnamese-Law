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
# Add this at the top of your file
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Store processed message IDs to prevent duplicates
processed_messages = set()
executor = ThreadPoolExecutor(max_workers=3)

@app.post("/webhook/")
async def webhook(request: Request):
    import uuid
    
    webhook_id = str(uuid.uuid4())[:8]
    print(f"[{webhook_id}] ==> Webhook messaging step started at {time.strftime('%H:%M:%S')}")
    
    try:
        chat_data = await request.json()
        print(f"[{webhook_id}] JSON parsed successfully")
        
        if chat_data.get("object") == "page":
            print(f"[{webhook_id}] Processing page object")
            
            for page_body in chat_data.get("entry", []):
                for message_obj in page_body.get("messaging", []):
                    if "message" in message_obj:
                        # Skip echo messages
                        if message_obj["message"].get("is_echo"):
                            print(f"[{webhook_id}] Skipping echo message")
                            continue
                        
                        # Get message ID to prevent duplicates
                        message_id = message_obj.get("message", {}).get("mid")
                        sender_id = message_obj.get("sender", {}).get("id")
                        
                        print(f"[{webhook_id}] Message ID: {message_id}, Sender: {sender_id}")
                        
                        # Check if we've already processed this message
                        if message_id in processed_messages:
                            print(f"[{webhook_id}] ⚠️ DUPLICATE MESSAGE - Already processed {message_id}")
                            continue
                        
                        # Mark as processed immediately
                        processed_messages.add(message_id)
                        print(f"[{webhook_id}] ✓ New message, marked as processed")
                        
                        # Clean up old processed messages (keep last 1000)
                        if len(processed_messages) > 1000:
                            processed_messages.clear()
                            print(f"[{webhook_id}] Cleared processed messages cache")
                        
                        # Process message asynchronously
                        asyncio.create_task(
                            process_message_background(webhook_id, message_obj, message_id)
                        )
            
            # Return immediately to Facebook (within 20 seconds)
            print(f"[{webhook_id}] <== Returning OK to Facebook")
            return {"status": "ok"}
        else:
            print(f"[{webhook_id}] Not a page object")
            raise HTTPException(status_code=400, detail="Not a page subscription")
            
    except Exception as e:
        print(f"[{webhook_id}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error"}  # Still return 200 to prevent retries

async def process_message_background(webhook_id: str, message_obj: dict, message_id: str):
    """Process message in background without blocking webhook response"""
    try:
        message = get_message(message_obj)
        sender_id = message_obj["sender"]["id"]
        
        print(f"[{webhook_id}] 🔄 Background processing: '{message}'")
        
        # Send typing indicator
        await send_typing_indicator(sender_id)
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            response = await loop.run_in_executor(
                executor, 
                lambda: inference(query=message).content
            )
            
            inference_time = time.time() - start_time
            print(f"[{webhook_id}] ✓ Inference completed in {inference_time:.2f}s")
            
            if isinstance(response, str) and response.strip():
                await send_message(recipient_id=sender_id, message=response)
                print(f"[{webhook_id}] ✓ Response sent successfully")
            else:
                await send_message(recipient_id=sender_id, message="Sorry, I couldn't generate a response.")
                print(f"[{webhook_id}] ⚠️ Empty response, sent error message")
                
        except Exception as inference_error:
            print(f"[{webhook_id}] ❌ Inference error: {inference_error}")
            await send_message(recipient_id=sender_id, message="Sorry, I'm having trouble processing that right now.")
            
    except Exception as e:
        print(f"[{webhook_id}] ❌ Background processing error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error message
        try:
            await send_message(
                recipient_id=message_obj["sender"]["id"],
                message="Oops! Something went wrong."
            )
        except:
            print(f"[{webhook_id}] ❌ Failed to send error message")

async def send_typing_indicator(recipient_id: str):
    """Send typing indicator to show bot is working"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                url="https://graph.facebook.com/v3.2/me/messages",
                params={"access_token": FB_ACCESS_TOKEN},
                json={
                    "recipient": {"id": recipient_id},
                    "sender_action": "typing_on"
                }
            )
            if response.status_code == 200:
                print("✓ Typing indicator sent")
            else:
                print(f"⚠️ Typing indicator failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Typing indicator error: {e}")

# Enhanced send_message with better error handling
async def send_message(recipient_id: str, message: str):
    print(f"📤 Sending to {recipient_id}: '{message[:50]}...'")
    
    MAX_MSG_LENGTH = 2000
    message_chunks = [message[i:i + MAX_MSG_LENGTH] for i in range(0, len(message), MAX_MSG_LENGTH)]

    async with httpx.AsyncClient(timeout=15.0) as client:
        for i, chunk in enumerate(message_chunks):
            try:
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
                    print(f"✅ Chunk {i+1}/{len(message_chunks)} sent successfully")
                else:
                    print(f"❌ Send failed: {response.status_code} - {response.text}")
                    break
                    
            except Exception as e:
                print(f"❌ Error sending chunk {i+1}: {e}")
                break

# Get message content
def get_message(message_obj):
    message = message_obj.get("message", {}).get("text", "")
    return message