#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from datetime import datetime as dt
from PIL import Image
import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.frames.frames import TransportMessageFrame
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TranscriptionFrame, InputImageRawFrame

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

video_participant_id = None
most_recent_img_frame = None


class TranscriptionFrameLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"!!! Transcription: {frame.text}")
        await self.push_frame(frame, direction)


class ImageConverter(FrameProcessor):
    async def process_frame(self, frame, direction):
        global most_recent_img_frame
        await super().process_frame(frame, direction)
        if isinstance(frame, InputImageRawFrame):
            # logger.debug("!!!")
            most_recent_img_frame = frame
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S_%f")
            # filename = f"cam-{timestamp}.png"
            filename = "cam.png"
            filepath = os.path.join("frames", filename)
            width, height = frame.size
            np_array = np.frombuffer(frame.image, dtype=np.uint8)
            img_array = np_array.reshape((height, width, 3))
            img = Image.fromarray(img_array, mode="RGB")
            img.save(filepath, "PNG")
        else:
            await self.push_frame(frame, direction)


async def move_forward(function_name, tool_call_id, arguments, llm, context, result_callback):
    distance = arguments["distance"]
    velocity = 40
    heading = 90
    angle = 0
    seconds = distance * 0.5

    await llm.push_frame(
        TransportMessageFrame({"message": f"move {velocity} {heading} {angle} {seconds}"})
    )
    await result_callback("Car is moving.")


async def move_backward(function_name, tool_call_id, arguments, llm, context, result_callback):
    distance = arguments["distance"]
    velocity = 40
    heading = 270
    angle = 0
    seconds = distance * 0.8

    await llm.push_frame(
        TransportMessageFrame({"message": f"move {velocity} {heading} {angle} {seconds}"})
    )
    await result_callback("Car is moving.")


# async def move_model_car(function_name, tool_call_id, arguments, llm, context, result_callback):
#     velocity = arguments["velocity"]
#     heading = arguments["heading"]
#     angle = arguments["angle"]
#     seconds = arguments["seconds"]

#     heading = heading + 90
#     await llm.push_frame(
#         TransportMessageFrame({"message": f"move {velocity} {heading} {angle} {seconds}"})
#     )
#     await result_callback("Car is moving.")


async def get_weather(function_name, tool_call_id, arguments, llm, context, result_callback):
    location = arguments["location"]
    await result_callback(f"The weather in {location} is currently 72 degrees and sunny.")


async def get_image(function_name, tool_call_id, arguments, llm, context, result_callback):
    question = arguments["question"]
    await llm.request_image_frame(user_id="*", text_content=question)
    await image_converter.push_frame(most_recent_img_frame)
    # await llm.push_frame(
    #     AnthropicImageMessageFrame(user_image_raw_frame=most_recent_img_frame, text=question)
    # )


async def main():
    global llm
    global image_converter

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                # transcription_enabled=True,
                vad_audio_passthrough=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        image_converter = ImageConverter()

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            # model="claude-3-5-sonnet-20240620",
            model="claude-3-5-sonnet-latest",
            enable_prompt_caching_beta=True,
        )
        llm.register_function("move_forward", move_forward)
        llm.register_function("move_backward", move_backward)
        # llm.register_function("move_model_car", move_model_car)
        llm.register_function("get_image", get_image)

        tools = [
            {
                "name": "move_forward",
                "description": "Move the model vehicle forward by the given distance in feet.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "distance": {
                            "type": "number",
                            "description": "Number of feet to move.",
                        },
                    },
                    "required": ["distance"],
                },
            },
            {
                "name": "move_backward",
                "description": "Move the model vehicle backward by the given distance in feet.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "distance": {
                            "type": "number",
                            "description": "Number of feet to move.",
                        },
                    },
                    "required": ["distance"],
                },
            },
            # {
            #     "name": "move_model_car",
            #     "description": "Move or turn a model vehicle. The vehicle will move at the specified velocity in the specified direction for the specified duration, while turning the specified number of radians.",
            #     "input_schema": {
            #         "type": "object",
            #         "properties": {
            #             "velocity": {
            #                 "type": "number",
            #                 "description": "How fast to move. Range is between 40 and 60.",
            #             },
            #             "heading": {
            #                 "type": "number",
            #                 "description": "The direction to move. 0 is forward. 180 is backward.",
            #             },
            #             "angle": {
            #                 "type": "number",
            #                 "description": "Number of pi radians to turn the car. Range is -1 to 1.",
            #             },
            #             "seconds": {
            #                 "type": "number",
            #                 "description": "Number of seconds to run the motors.",
            #             },
            #         },
            #         "required": ["location"],
            #     },
            # },
            {
                "name": "get_image",
                "description": "Get an image from the video stream.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question that the user is asking about the image.",
                        }
                    },
                    "required": ["question"],
                },
            },
        ]

        # todo: test with very short initial user message

        system_prompt = """\
You are a model car named SupaRover. You can talk to the user. The user can ask you to move.

You always keep your responses concise. Extremely concise. Blunt even. You do NOT give long wordy, overly friendly responses.

You can move forward, backward, left, or right. You can also turn left or right.

You have access to several functions to call to move. For example, call move_forward with a specified distance to move the car forward.

When the user asks you to move, call the appropriate function with the appropriate parameters.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that \
indicate you should use the get_image tool are:
  - What do you see?
  - What's in the video?
  - Can you describe the video?
  - Tell me about what you see.
  - Tell me something interesting about what you see.
  - What's happening in the video?

If you need to use a tool, simply use the tool. Do not tell the user the tool you are using. Be brief and concise.
 
Start by introducing yourself. 
"""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {"role": "user", "content": "Start the conversation by introducing yourself."},
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                image_converter,
                stt,
                TranscriptionFrameLogger(),
                context_aggregator.user(),  # User speech to text
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses and tool context
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            print(f"First participant joined: {participant}")
            # global video_participant_id
            # video_participant_id = participant["id"]
            # await transport.capture_participant_transcription(video_participant_id)
            # await transport.capture_participant_video(video_participant_id, framerate=0)
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            print(f"Participant joined -- setting up video capture: {participant}")
            await transport.capture_participant_video(participant["id"], 1)

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
