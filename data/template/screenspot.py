# _SCREENSPOT_SYSTEM = """You are a computer assistant trained to navigate digital screens. 
# Given an instruction and a screen observation, 
# output the location [x,y] to click the element. 
# Positions should be the relative coordinates on the screenshot, scaled to a range of 0-1.
# """

# _SCREENSPOT_USER = """{system}
# Instruction: {element}
# Observation: <|image_1|>
# What is the position?
# """

_SCREENSPOT_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
_SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
_SYSTEM_point_int = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 1 to 1000."

_SCREENSPOT_USER = '<|image_1|>{system}{element}'

def screenspot_to_openai(element_name, answer_xy=None, xy_int=False):
    transformed_data = []
    if xy_int:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point_int
    else:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point

    question = _SCREENSPOT_USER.format(system=system_prompt, element=element_name)
    transformed_data.append(
                {
                    "role": "user",
                    "content": question,
                },
            )

    if answer_xy:
        answer = f'{answer_xy}'
        transformed_data.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                )
    return transformed_data

def screenspot_to_openai_qwen(element_name, image, xy_int=False):
    transformed_data = []
    user_content = []

    if xy_int:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point_int
    else:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point

    '{system}<|image_1|>{element}'
    user_content.append({"type": "text", "text": system_prompt})
    user_content.append(image)
    user_content.append({"type": "text",  "text": element_name})

    # question = _SCREENSPOT_USER.format(system=_SCREENSPOT_SYSTEM, element=element_name)
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data

# short prompt
# _SYSTEM = "In the UI, where should I click if I want to complete instruction \"{}\"?"
# _USER = '{system}<|image_1|>'

# def screenspot_to_openai(element_name, answer_xy=None):
#     transformed_data = []
#     system_prompt = _SYSTEM.format(element_name)
#     question = _USER.format(system=system_prompt)
#     transformed_data.append(
#                 {
#                     "role": "user",
#                     "content": question,
#                 },
#             )

#     if answer_xy:
#         answer = f'{answer_xy}'
#         transformed_data.append(
#                     {
#                         "role": "assistant",
#                         "content": answer,
#                     },
#                 )
#     return transformed_data