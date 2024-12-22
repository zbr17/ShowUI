from .xlam import xlam_to_openai
from .miniwob import miniwob_to_openai, miniwob_to_openai_qwenvl
from .mind2web import mind2web_to_openai, mind2web_to_openai_qwenvl
from .aitw import aitw_to_openai, aitw_to_openai_qwenvl
from .aitz import aitz_to_openai, aitz_to_openai_qwenvl
from .screenspot import screenspot_to_openai, screenspot_to_openai_qwen
from .seeclick import seeclick_to_openai
from .act2cap import act2cap_to_openai

from .odyssey import odyssey_to_openai, odyssey_to_openai_qwenvl
from .guiworld import guiworld_to_openai
from .omniact import omniact_to_openai, omniact_to_openai_qwenvl

from .shared_grounding import grounding_to_openai, grounding_to_openai_qwen
from .shared_onestep import onestep_to_openai, onestep_to_openai_qwen
from .shared_captioning import captioning_to_openai, captioning_to_openai_qwen
from .shared_navigation import navigation_to_openai, navigation_to_openai_qwenvl
from .shared_chat import chat_to_openai, chat_to_openai_qwen
from .shared_llava import llava_to_openai, llava_to_openai_qwen

from .utils import batch_add_answer, batch_add_answer_append