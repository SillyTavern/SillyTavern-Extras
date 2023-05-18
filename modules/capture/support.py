from flask import jsonify
from json import loads
from datetime import datetime
from PIL import ImageFont, Image, ImageDraw

# GNU Unifont, replace if find better font with lots of unicode support
FONT = "./modules/capture/unifont-15.0.01.ttf"


# incase this ever changes
DATE_FORMAT = "%Y-%m-%d @%Hh %Mm %Ss %fms"

# make this a parameter later?
WIDTH = 1024


class Message:
    def __init__(self, name: str, is_user: bool, timestamp: str, message) -> None:
        self.name = name
        self.is_user = is_user
        self.timestamp = self.to_timestamp(timestamp)
        self.message = message

    def to_timestamp(self, timestamp: str) -> datetime:
        return datetime.strptime(timestamp, DATE_FORMAT)


class Chat:
    def __init__(self, messages=[]) -> None:
        self.messages = messages

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def export(self) -> str:
        return [ob.__dict__ for ob in self.messages]


class Capture:
    def __init__(self, jsonl: str, pfp_user=None, pfp_char=None) -> None:
        self.jsonl = jsonl
        self.jsonl_to_enumerable()
        self.extract_data()
        self.font_size = 16
        self.bold_font_size = 18  # Increased font size for the name
        self.line_spacing = 4
        self.font = ImageFont.truetype(FONT, self.font_size)
        self.bold_font = ImageFont.truetype(
            FONT, self.bold_font_size
        )  # Bold font variant
        self.width = WIDTH
        self.pfp_user = Image.open(pfp_user) if pfp_user else None
        self.pfp_char = Image.open(pfp_char) if pfp_char else None
        self.profile_pic_size = (40, 40)
        self.resize_pfps()
        self.calculate_chat_height()
        self.prepare_image()

    def jsonl_to_enumerable(self):
        self.data = (loads(line) for line in self.jsonl.splitlines())

    def extract_data(self):
        chat = Chat()
        for line in self.data:
            if "chat_metadata" in line:
                continue
            else:
                chat.add_message(
                    Message(
                        line["name"],
                        line["is_user"],
                        line["send_date"],
                        line["mes"],
                    )
                )

        self.chat = chat

    def calculate_chat_height(self, padding=10):
        total_height = padding
        self.message_heights = []

        for message in self.chat.messages:
            text = message.message
            wrapped_text = self.wrap_text(text)
            wrapped_text_height = len(wrapped_text) * (
                self.font.getsize(" ")[1] + self.line_spacing
            )
            total_height += wrapped_text_height

            # Add additional spacing for the name
            total_height += self.bold_font.getsize(" ")[1] + self.line_spacing

            # Add separation line height
            total_height += self.line_spacing

            # Store the height of each message separately
            message_height = (
                wrapped_text_height
                + self.bold_font.getsize(" ")[1]
                + 2 * self.line_spacing
            )
            self.message_heights.append(message_height)

        # Calculate the profile picture heights
        total_height += len(self.chat.messages) * self.line_spacing

        # Add padding at the top and bottom
        total_height += padding * 2

        self.height = total_height

    def wrap_text(self, text, padding=10):
        words = text.split()
        lines = []
        current_line = words[0]

        for word in words[1:]:
            line_width, _ = self.font.getsize(current_line + " " + word)
            if line_width <= self.width - self.profile_pic_size[0] - padding * 2:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)
        return lines

    def resize_pfps(self):
        self.pfp_user.thumbnail(self.profile_pic_size)
        self.pfp_char.thumbnail(self.profile_pic_size)

    def prepare_image(self, padding=10):
        height = self.height + padding * 2
        width = self.width + padding * 2
        self.image = Image.new("RGB", (width, height), color="white")
        self.draw = ImageDraw.Draw(self.image)

        y = padding
        for message in self.chat.messages:
            text = message.message
            wrapped_text = self.wrap_text(text)

            # Draw profile picture
            if message.is_user:
                profile_pic = self.pfp_user
            else:
                profile_pic = self.pfp_char

            self.image.paste(profile_pic, (padding, y))

            # Draw name
            name = message.name
            name_x = padding + self.profile_pic_size[0] + padding
            name_y = y
            self.draw.text(
                (name_x, name_y),
                name,
                fill="black",
                font=self.bold_font,  # Use a bolder font variant or increase the font size
                spacing=self.line_spacing,
            )

            # Increase y position
            y += (
                self.bold_font.getsize(" ")[1] + self.line_spacing
            )  # Use the bold font's size

            # Draw message text
            x = padding + self.profile_pic_size[0] + padding
            for line in wrapped_text:
                self.draw.text(
                    (x, y),
                    line,
                    fill="black",
                    font=self.font,
                    spacing=self.line_spacing,
                )
                y += self.font.getsize(" ")[1] + self.line_spacing

            # Draw separation line
            line_y = y + self.line_spacing
            line_start = (padding, line_y)
            line_end = (width - padding, line_y)
            self.draw.line([line_start, line_end], fill="black")

            # Increase y position
            y = line_y + self.line_spacing

        return self.image
