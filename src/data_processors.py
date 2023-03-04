import sys
from glob import glob
import re

from bs4 import BeautifulSoup
import emoji
from tqdm import tqdm

from src.data_utils import *

replace_table = {"ã¥": "å", "ã¶": "ö", "ã¤": "ä", "\n": "\t", "ð": ""}
NOPRINT_TRANS_TABLE = {i: " " for i in range(0, sys.maxunicode + 1) if not (chr(i).isprintable())}


def replace_utf_stuff(s: str):
    s = emoji.replace_emoji(s, ' ')
    for k, v in replace_table.items():
        s = s.replace(k, v)
    s = s.translate(NOPRINT_TRANS_TABLE)
    return s.replace("  ", " ")


@cache_result("messenger_data.json")
def process_messenger():
    MESSENGER_PATH = str(DATA_SOURCES / "messenger" / "*")
    MESSENGER_USERS = glob(MESSENGER_PATH)
    messages_jsons = []
    for msg_user in MESSENGER_USERS:
        for msg_json in glob(str(Path(msg_user) / "message_*.json")):
            messages_jsons.append(msg_json)

    messages_jsons = list(map(load, tqdm(messages_jsons)))

    extract_name = "Hugo Berg"
    extracted_msgs = []
    for msg_data in messages_jsons:
        extracted_msgs.extend(list(map(lambda c: replace_utf_stuff(c.lower()), filter(lambda x: x is not None,
                                                                                      [x.get("content", None) for x in
                                                                                       msg_data["messages"] if x[
                                                                                           "sender_name"] == extract_name]))))
    return extracted_msgs


@cache_result("whatsapp_data.json")
def process_whatsapp():
    WHATSAPP_PATH = str(DATA_SOURCES / "whatsapp" / "WhatsApp Chat with *.txt")
    WHATSAPP_FILES = glob(WHATSAPP_PATH)
    chatlogs = []
    for wa_file in WHATSAPP_FILES:
        with open(wa_file, mode="r", encoding="utf-8") as cl_file:
            chatlogs.append(cl_file.read())

    whatsapp_pattern = r"\d{4}-\d{2}-\d{2}, \d{2}:\d{2} - "
    whatsapp_re = re.compile(whatsapp_pattern)
    full_chatlogs = "\n".join(chatlogs)
    split_messages = re.split(whatsapp_re, full_chatlogs)
    extract_user = "Hugo Berg:"
    extracted_messages = [x[len(extract_user):].strip().lower() for x in split_messages if x.startswith(extract_user)]
    return list(map(replace_utf_stuff, extracted_messages))


@cache_result("instagram_data.json")
def process_instagram():
    IG_PATH = str(DATA_SOURCES / "instagram" / "*" / "message_*.html")
    IG_FILES = glob(IG_PATH)
    parsed_htmls = []
    for ig_file in tqdm(IG_FILES):
        with open(ig_file, mode="r", encoding="utf-8") as ig_f:
            parsed_htmls.append(BeautifulSoup(ig_f.read(), "html.parser"))

    extract_person = "hugo berg"
    all_messages = []
    for obj in parsed_htmls:
        relevant_html_objs = [x for x in obj.body.find_all("div", attrs={"class": "_a6-p"}) if
                              x.find_parent().find("div",
                                                   attrs={"class": "_a6-i"}).get_text().lower() == extract_person]
        extract_messages = [x.get_text().strip().lower() for x in relevant_html_objs]

        def filter_msg_by(s: str) -> bool:
            if s.startswith("❤️"): return False
            if s == "": return False
            if s == "liked a message": return False
            if "www.instagram.com" in s: return False
            return True

        def remove_liked_msg(s: str) -> str:
            if "❤️" in s:
                s = s[:s.rindex("❤️")]
            return replace_utf_stuff(s).strip()

        extract_messages = list(map(remove_liked_msg, filter(filter_msg_by, extract_messages)))
        extract_messages = [x for x in extract_messages if x]
        all_messages.extend(extract_messages)
    return all_messages


@cache_result("discord_data.json")
def process_discord():
    DISCORD_PATH = DATA_SOURCES / "discord" / "*.json"
    discord_files = glob(str(DISCORD_PATH))
    msg_datas = []
    for f in tqdm(discord_files):
        msg_datas.append(load(f)["messages"])
    msgs = []
    for msg_data in msg_datas:
        for msg in msg_data:
            msgs.append(replace_utf_stuff(msg["content"]).strip())
    msgs = [x for x in msgs if x]
    return msgs


def process_all():
    all_text = []
    for f in (process_discord, process_instagram, process_whatsapp, process_messenger):
        all_text.extend(f())
    return all_text


@cache_result("all_text.json")
def store_proced_data(savename: str="input.txt"):
    all_msgs = process_all()
    write_txt = "\n".join(all_msgs)
    with open(PROCED_DATA / savename, mode="w") as f:
        f.write(write_txt)
    return write_txt
