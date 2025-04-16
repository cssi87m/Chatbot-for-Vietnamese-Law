from typing import Dict, List, Optional, Union, Any

from collections import defaultdict
from langchain_core.messages import BaseMessage
import json

class JsonChatStorage:
    def __init__(self, max_history_size: Optional[int] = None):
        self.conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history_size = max_history_size

        self.file_name = 'data.json'

        # try:
        #     with open(self.file_name, 'r') as f:
        #         self.conversations = json.load(f)
        # except:
        #     pass
    
    def trim_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.max_history_size is None:
            return conversation
        else:
            return conversation[-self.max_history_size:]
    def save_chat_message(
        self,
        user_id: str,
        new_message: BaseMessage,
    ) -> bool:
        try:

            self.conversations[user_id].append(new_message.to_dict())

            self.conversations[user_id] = self.trim_conversation(self.conversations[user_id])

            with open(self.file_name, 'w') as f:
                json.dump(self.conversations, f, indent=4)
            
            return True
        except:
            return False


#    def fetch_chat(
#         self,
#         user_id: str,
#         session_id: str,
#         agent_id: str,
#     ) -> List[ConversationMessage]:
#         key = self._generate_key(user_id, session_id, agent_id)
#         conversation = self.conversations[key]
#         conversation = self.trim_conversation(conversation)
#         ret_conversation = [ConversationMessage(**message) for message in conversation]
#         return ret_conversation

#     def fetch_all_chats(
#         self,
#         user_id: str,
#         session_id: str
#     ) -> Dict[str, List[ConversationMessage]]:
#         all_messages: Dict[str, List[ConversationMessage]] = {}
#         for key, messages in self.conversations.items():
#             stored_user_id, stored_session_id, agent_id = key.split('#')
#             if stored_user_id == user_id and stored_session_id == session_id:
#                 all_messages[self._generate_key(user_id, session_id, agent_id)] = [ConversationMessage(**message) for message in messages]

#         return all_messages
