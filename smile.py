#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================== SMILE PARTY BOT - ULTIMATE QR SYSTEM ====================

import warnings
warnings.filterwarnings("ignore", message="If 'per_message=False'")

import json
import re
import logging
import logging.handlers
import asyncio
import sqlite3
import random
import string
import shutil
import os
import time
import csv
import html
import hashlib
import hmac
import base64
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import closing
import traceback
import tempfile
import threading
import io
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

# QR Code libraries
import qrcode
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np

# For QR scanning
try:
    import cv2
    from pyzbar.pyzbar import decode
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    decode = None
    CV2_AVAILABLE = False
    print("⚠️ OpenCV не установлен. Используется базовое распознавание QR-кодов.")
    print("   Для установки: pip install opencv-python pyzbar")

# For caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis не установлен. Используется файловое кэширование QR-кодов.")
    print("   Для установки: pip install redis")

# ========== НАСТРОЙКИ БОТА ==========
BOT_TOKEN = "8433063885:AAFPT2fYk6HQB1gt-x2kxqaIaSJE9U3tQdM"
ADMIN_IDS = [7978634199, 1037472337, 932339331]
PROMOTER_IDS = [7283583682, 6179688188, 932339331, 8387903981, 8041100755, 1380285963, 1991277474, 8175354320, 6470777539, 8470198654, 7283630429, 8396505232, 8176926325, 8566108065, 7978634199, 1037472337]
SCANNER_IDS = list(set(ADMIN_IDS + PROMOTER_IDS))

# ID каналов и чатов
CLOSED_ORDERS_CHANNEL_ID = -1003780187586
REFUND_ORDERS_CHANNEL_ID = -1003735636374
PROMOTERS_CHAT_ID = -1003105307057
LISTS_CHANNEL_ID = -1003661551964
LOGS_CHANNEL_ID = -1003610531501

# Файл базы данных
DB_FILE = "smile_party_bot.db"

# ========== НАСТРОЙКИ QR-КОДОВ ==========
QR_CONFIG = {
    "secret_key": "smile_party_super_secret_key_2024_CHANGE_ME",
    "version": "1.0",
    "cache_dir": "qr_cache",
    "cache_ttl": 86400,
    "qr_size": 10,
    "logo_path": None,
    "enable_watermark": True,
    "watermark_text": "SMILE PARTY",
    "max_scan_attempts": 3,
    "scan_timeout": 60,
    "offline_mode": False,
    "enable_hmac": True,
    "enable_timestamp": True,
    "enable_qr_caching": True
}

# ========== НАСТРОЙКИ ТИПОВ БИЛЕТОВ ==========
TICKET_TYPES = {
    "standard": {
        "name": "Танцпол 🎟",
        "price_standard": 450,
        "price_group": 350
    },
    "vip": {
        "name": "VIP 🎩",
        "price": 650
    }
}

# ========== НАСТРОЙКА РАСШИРЕННОГО ЛОГИРОВАНИЯ ==========
def setup_advanced_logging():
    import sys
    import io
    
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    if sys.platform == "win32":
        import codecs
        console_handler.stream = io.TextIOWrapper(
            console_handler.stream.buffer,
            encoding='utf-8',
            errors='ignore'
        )
    
    logger.addHandler(console_handler)
    
    file_handler = logging.handlers.RotatingFileHandler(
        'bot.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    user_logger = logging.getLogger('user_actions')
    user_handler = logging.handlers.RotatingFileHandler(
        'user_actions.log',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    user_handler.setFormatter(formatter)
    user_logger.addHandler(user_handler)
    
    qr_logger = logging.getLogger('qr_codes')
    qr_handler = logging.handlers.RotatingFileHandler(
        'qr_codes.log',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    qr_handler.setFormatter(formatter)
    qr_logger.addHandler(qr_handler)
    
    performance_logger = logging.getLogger('performance')
    perf_handler = logging.handlers.RotatingFileHandler(
        'performance.log',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    perf_handler.setFormatter(formatter)
    performance_logger.addHandler(perf_handler)
    
    return logger, user_logger, qr_logger, performance_logger

logger, user_logger, qr_logger, perf_logger = setup_advanced_logging()

# ========== QR CODE MANAGER ==========
class QRCodeManager:
    def __init__(self, config: Dict = None):
        self.config = config or QR_CONFIG
        self.stats = defaultdict(int)
        self.stats_lock = threading.Lock()
        self.cache = {}
        self.last_scan = defaultdict(float)
        
        if self.config["enable_qr_caching"]:
            os.makedirs(self.config["cache_dir"], exist_ok=True)
            logger.info(f"📁 Директория кэша QR-кодов: {self.config['cache_dir']}")
        
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=False,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                logger.info("✅ Redis подключен для кэширования QR-кодов")
            except:
                self.redis_client = None
                logger.warning("⚠️ Redis недоступен, используется файловое кэширование")
        
        logger.info("🚀 QR Code Manager инициализирован")
    
    def _generate_hmac(self, data: str) -> str:
        if not self.config["enable_hmac"]:
            return ""
        
        message = data.encode('utf-8')
        signature = hmac.new(
            self.config["secret_key"].encode('utf-8'),
            message,
            hashlib.sha256
        ).hexdigest()[:8]
        return signature
    
    def _verify_hmac(self, data: str, signature: str) -> bool:
        if not self.config["enable_hmac"]:
            return True
        
        expected = self._generate_hmac(data)
        return hmac.compare_digest(expected, signature)
    
    def _add_timestamp(self, data: str) -> str:
        if not self.config["enable_timestamp"]:
            return data
        
        timestamp = int(time.time())
        return f"{data}|{timestamp}"
    
    def _verify_timestamp(self, data: str, max_age: int = 86400) -> Tuple[bool, str]:
        if '|' not in data or not self.config["enable_timestamp"]:
            return True, data
        
        try:
            base_data, timestamp_str = data.rsplit('|', 1)
            timestamp = int(timestamp_str)
            current_time = int(time.time())
            
            if current_time - timestamp > max_age:
                return False, base_data
            
            return True, base_data
        except:
            return False, data
    
    def prepare_qr_data(self, order_code: str, ticket_type: str = "standard", guest_name: str = "") -> str:
        base_data = f"SMILE_PARTY:{order_code}:{ticket_type}"
        if guest_name:
            guest_hash = hashlib.md5(guest_name.encode()).hexdigest()[:8]
            base_data += f":{guest_hash}"
        
        base_data = f"V{self.config['version']}:{base_data}"
        data_with_time = self._add_timestamp(base_data)
        signature = self._generate_hmac(data_with_time)
        
        return f"{data_with_time}|{signature}"
    
    def parse_qr_data(self, qr_data: str) -> Dict:
        result = {
            "valid": False,
            "code": None,
            "ticket_type": None,
            "guest_hash": None,
            "error": None,
            "data": qr_data
        }
        
        try:
            parts = qr_data.split('|')
            if len(parts) < 2:
                result["error"] = "Неверный формат данных"
                return result
            
            data_part = '|'.join(parts[:-1])
            signature = parts[-1]
            
            if not self._verify_hmac(data_part, signature):
                result["error"] = "Недействительная подпись"
                return result
            
            timestamp_valid, data_without_time = self._verify_timestamp(data_part)
            if not timestamp_valid:
                result["error"] = "Истек срок действия QR-кода"
                return result
            
            main_parts = data_without_time.split(':')
            if len(main_parts) < 3:
                result["error"] = "Неверная структура данных"
                return result
            
            version = main_parts[0]
            prefix = main_parts[1]
            
            if not version.startswith('V') or prefix != "SMILE_PARTY":
                result["error"] = "Неизвестный формат QR-кода"
                return result
            
            code = main_parts[2]
            ticket_type = main_parts[3] if len(main_parts) > 3 else "standard"
            guest_hash = main_parts[4] if len(main_parts) > 4 else ""
            
            result.update({
                "valid": True,
                "code": code,
                "ticket_type": ticket_type,
                "guest_hash": guest_hash,
                "version": version
            })
            
        except Exception as e:
            result["error"] = f"Ошибка парсинга: {str(e)}"
        
        return result
    
    def generate_qr_image(self, data: str, ticket_type: str = "standard", guest_name: str = "") -> bytes:
        start_time = time.time()
        
        cache_key = hashlib.md5(f"{data}_{ticket_type}_{guest_name}".encode()).hexdigest()
        
        cached = self._get_from_cache(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            logger.debug(f"✅ QR-код получен из кэша: {cache_key[:8]}")
            perf_logger.info(f"QR_GEN_CACHE_HIT,{cache_key[:8]},{time.time()-start_time:.3f}")
            return cached
        
        self.stats["cache_misses"] += 1
        
        try:
            logger.info(f"🚀 Генерация QR-кода для: {data[:30]}...")
            
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=self.config["qr_size"],
                border=4,
            )
            
            prepared_data = self.prepare_qr_data(data, ticket_type, guest_name)
            qr.add_data(prepared_data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            
            if self.config["logo_path"] and os.path.exists(self.config["logo_path"]):
                img = self._add_logo(img)
            
            if self.config["enable_watermark"]:
                img = self._add_watermark(img, self.config["watermark_text"])
            
            img = self._add_styling(img, data, ticket_type, guest_name)
            
            img_bytes = self._image_to_bytes(img)
            
            self._save_to_cache(cache_key, img_bytes)
            
            with self.stats_lock:
                self.stats["qr_generated"] += 1
                self.stats["total_generation_time"] += time.time() - start_time
            
            logger.info(f"✅ QR-код сгенерирован за {time.time()-start_time:.2f}с")
            perf_logger.info(f"QR_GEN_SUCCESS,{cache_key[:8]},{time.time()-start_time:.3f}")
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации QR-кода: {e}")
            logger.error(traceback.format_exc())
            
            with self.stats_lock:
                self.stats["qr_errors"] += 1
            
            perf_logger.info(f"QR_GEN_ERROR,{cache_key[:8]},{str(e)[:50]}")
            
            return self._generate_fallback_qr(data)
    
    def _add_logo(self, img: Image.Image) -> Image.Image:
        try:
            logo = Image.open(self.config["logo_path"])
            
            qr_width, qr_height = img.size
            logo_size = int(qr_width * 0.2)
            
            logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            
            mask = Image.new('L', (logo_size, logo_size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, logo_size, logo_size), fill=255)
            
            pos = ((qr_width - logo_size) // 2, (qr_height - logo_size) // 2)
            img.paste(logo, pos, mask)
            
            logger.debug("✅ Логотип добавлен в QR-код")
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления логотипа: {e}")
        
        return img
    
    def _add_watermark(self, img: Image.Image, text: str) -> Image.Image:
        try:
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            width, height = img.size
            
            for i in range(0, width, 100):
                for j in range(0, height, 100):
                    txt_img = Image.new('RGBA', img.size, (255,255,255,0))
                    txt_draw = ImageDraw.Draw(txt_img)
                    txt_draw.text((i, j), text, fill=(128,128,128,30), font=font)
                    
                    img = Image.alpha_composite(img.convert('RGBA'), txt_img)
            
            logger.debug("✅ Водяной знак добавлен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления водяного знака: {e}")
        
        return img
    
    def _add_styling(self, img: Image.Image, data: str, ticket_type: str, guest_name: str) -> Image.Image:
        try:
            width, height = img.size
            new_height = height + 60
            
            new_img = Image.new('RGB', (width, new_height), 'white')
            new_img.paste(img, (0, 0))
            
            draw = ImageDraw.Draw(new_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            ticket_type_text = "VIP" if ticket_type == "vip" else "STANDARD"
            display_text = f"#{data} | {ticket_type_text}"
            if guest_name:
                display_text += f" | {guest_name[:20]}"
            
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            text_y = height + 10
            
            draw.text((text_x+2, text_y+2), display_text, fill="gray", font=font)
            draw.text((text_x, text_y), display_text, fill="black", font=font)
            
            draw.rectangle([(0, 0), (width-1, height-1)], outline="black", width=1)
            
            logger.debug("✅ Стилизация QR-кода завершена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка стилизации: {e}")
        
        return new_img
    
    def _image_to_bytes(self, img: Image.Image) -> bytes:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', optimize=True)
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    def _generate_fallback_qr(self, data: str) -> bytes:
        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=4)
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            logger.info("✅ Создан fallback QR-код")
            return img_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка fallback QR: {e}")
            img = Image.new('RGB', (200, 200), 'white')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            return img_bytes.getvalue()
    
    def _get_from_cache(self, key: str) -> Optional[bytes]:
        if not self.config["enable_qr_caching"]:
            return None
        
        if self.redis_client:
            try:
                data = self.redis_client.get(f"qr:{key}")
                if data:
                    return data
            except:
                pass
        
        cache_path = os.path.join(self.config["cache_dir"], f"{key}.png")
        if os.path.exists(cache_path):
            if time.time() - os.path.getmtime(cache_path) < self.config["cache_ttl"]:
                with open(cache_path, 'rb') as f:
                    return f.read()
            else:
                os.remove(cache_path)
        
        return None
    
    def _save_to_cache(self, key: str, data: bytes):
        if not self.config["enable_qr_caching"]:
            return
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"qr:{key}",
                    self.config["cache_ttl"],
                    data
                )
                return
            except:
                pass
        
        try:
            cache_path = os.path.join(self.config["cache_dir"], f"{key}.png")
            with open(cache_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в кэш: {e}")
    
    def scan_qr_image(self, image_bytes: bytes) -> Dict:
        start_time = time.time()
        
        result = {
            "success": False,
            "data": None,
            "parsed": None,
            "error": None,
            "scan_time": 0
        }
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                result["error"] = "Не удалось прочитать изображение"
                return result
            
            img = self._enhance_image_for_scan(img)
            
            qr_data = self._decode_qr_multiple_methods(img)
            
            if qr_data:
                result["success"] = True
                result["data"] = qr_data
                result["parsed"] = self.parse_qr_data(qr_data)
                
                with self.stats_lock:
                    self.stats["qr_scanned"] += 1
                
                logger.info(f"✅ QR-код распознан: {qr_data[:30]}...")
            else:
                result["error"] = "QR-код не найден на изображении"
                with self.stats_lock:
                    self.stats["scan_failures"] += 1
            
            result["scan_time"] = time.time() - start_time
            perf_logger.info(f"QR_SCAN,{result['success']},{result['scan_time']:.3f}")
            
        except Exception as e:
            result["error"] = f"Ошибка сканирования: {str(e)}"
            logger.error(f"❌ Ошибка сканирования QR: {e}")
            
            with self.stats_lock:
                self.stats["scan_errors"] += 1
        
        return result
    
    def _enhance_image_for_scan(self, img) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            gray = cv2.equalizeHist(gray)
            
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            denoised = cv2.medianBlur(binary, 3)
            
            height, width = denoised.shape
            if width < 300 or height < 300:
                scale = max(300 / width, 300 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                denoised = cv2.resize(denoised, (new_width, new_height))
            
            return denoised
            
        except Exception as e:
            logger.error(f"❌ Ошибка улучшения изображения: {e}")
            return img
    
    def _decode_qr_multiple_methods(self, img) -> Optional[str]:
        if CV2_AVAILABLE and decode is not None:
            try:
                decoded_objects = decode(img)
                if decoded_objects:
                    return decoded_objects[0].data.decode('utf-8')
            except Exception as e:
                logger.debug(f"Pyzbar ошибка: {e}")
        
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(img)
            if retval and decoded_info and decoded_info[0]:
                return decoded_info[0]
        except Exception as e:
            logger.debug(f"OpenCV QR detector ошибка: {e}")
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if CV2_AVAILABLE and decode is not None:
                decoded_objects = decode(thresh)
                if decoded_objects:
                    return decoded_objects[0].data.decode('utf-8')
        except Exception as e:
            logger.debug(f"Otsu метод ошибка: {e}")
        
        return None
    
    def check_scan_rate_limit(self, scanner_id: int, order_code: str) -> Tuple[bool, int]:
        key = f"{scanner_id}:{order_code}"
        current_time = time.time()
        
        if key in self.last_scan:
            time_diff = current_time - self.last_scan[key]
            if time_diff < self.config["scan_timeout"]:
                return False, int(self.config["scan_timeout"] - time_diff)
        
        self.last_scan[key] = current_time
        return True, 0
    
    def get_stats(self) -> Dict:
        with self.stats_lock:
            stats = dict(self.stats)
            stats["cache_hit_rate"] = 0
            if stats.get("cache_hits", 0) + stats.get("cache_misses", 0) > 0:
                total = stats.get("cache_hits", 0) + stats.get("cache_misses", 0)
                stats["cache_hit_rate"] = (stats.get("cache_hits", 0) / total) * 100
            
            if stats.get("qr_generated", 0) > 0:
                stats["avg_generation_time"] = (
                    stats.get("total_generation_time", 0) / stats.get("qr_generated", 1)
                )
            
            return stats
    
    def clear_cache(self, older_than: int = None) -> int:
        cleared = 0
        
        if self.redis_client:
            try:
                pass
            except:
                pass
        
        cache_dir = self.config["cache_dir"]
        if os.path.exists(cache_dir):
            current_time = time.time()
            for filename in os.listdir(cache_dir):
                if filename.endswith('.png'):
                    filepath = os.path.join(cache_dir, filename)
                    file_age = current_time - os.path.getmtime(filepath)
                    
                    if older_than is None or file_age > older_than:
                        os.remove(filepath)
                        cleared += 1
        
        logger.info(f"🧹 Очищено {cleared} файлов из кэша")
        return cleared

qr_manager = QRCodeManager()

from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    ReplyKeyboardRemove
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
    ApplicationBuilder
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, TelegramError

class RateLimiter:
    def __init__(self, max_calls: int = 10, time_window: int = 5):
        self.user_requests = {}
        self.max_calls = max_calls
        self.time_window = time_window
    
    def check_limit(self, user_id: int) -> bool:
        current_time = time.time()
        
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if current_time - req_time < self.time_window
        ]
        
        if len(self.user_requests[user_id]) >= self.max_calls:
            return False
        
        self.user_requests[user_id].append(current_time)
        return True
    
    def get_remaining(self, user_id: int) -> int:
        current_time = time.time()
        
        if user_id not in self.user_requests:
            return self.max_calls
        
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if current_time - req_time < self.time_window
        ]
        
        return self.max_calls - len(self.user_requests[user_id])

rate_limiter = RateLimiter(max_calls=15, time_window=5)

def sanitize_input(text: str, max_length: int = 500) -> str:
    if not text:
        return ""
    
    text = html.escape(text)
    
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_name(name: str) -> bool:
    if len(name) < 2 or len(name) > 100:
        return False
    
    pattern = r'^[a-zA-Zа-яА-ЯёЁ\s\-\'\.]+$'
    return bool(re.match(pattern, name))

async def send_log_to_channel(context: ContextTypes.DEFAULT_TYPE, message: str, level: str = "INFO"):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        if len(log_message) > 4000:
            log_message = log_message[:4000] + "..."
        
        await context.bot.send_message(
            chat_id=LOGS_CHANNEL_ID,
            text=f"`{log_message}`",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Ошибка отправки лога в канал: {e}")

def log_user_action(user_id: int, action: str, details: str = ""):
    try:
        user_logger.info(f"User {user_id} - {action} - {details}")
    except Exception as e:
        logger.error(f"Ошибка логирования действия пользователя: {e}")

def log_qr_action(action: str, details: Dict = None):
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {}
        }
        qr_logger.info(json.dumps(log_entry, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Ошибка логирования QR-действия: {e}")

def generate_unique_code(length: int = 6) -> str:
    characters = string.digits
    while True:
        numbers = ''.join(random.choices(characters, k=length))
        code = f"#KA{numbers}"
        return code

def format_code_for_display(code: str) -> str:
    return code

class Database:
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.init_database()
        self.check_and_fix_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        with closing(self.get_connection()) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key VARCHAR(50) UNIQUE NOT NULL,
                    setting_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id BIGINT UNIQUE NOT NULL,
                    username VARCHAR(100),
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    role VARCHAR(20) DEFAULT 'user',
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    notified_about_restart BOOLEAN DEFAULT FALSE,
                    request_count INTEGER DEFAULT 0,
                    last_request TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id VARCHAR(20) UNIQUE NOT NULL,
                    order_code VARCHAR(20) UNIQUE NOT NULL,
                    user_id BIGINT NOT NULL,
                    username VARCHAR(100),
                    user_name VARCHAR(200) NOT NULL,
                    user_email VARCHAR(100) NOT NULL,
                    group_size INTEGER NOT NULL,
                    ticket_type VARCHAR(10) DEFAULT 'standard',
                    total_amount INTEGER NOT NULL,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    assigned_promoter VARCHAR(100),
                    closed_by VARCHAR(100),
                    closed_at TIMESTAMP,
                    notified_promoters BOOLEAN DEFAULT FALSE,
                    processed_at TIMESTAMP,
                    scanned_at TIMESTAMP,
                    scanned_by VARCHAR(100),
                    qr_hash VARCHAR(64),
                    qr_version VARCHAR(10)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS guests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id VARCHAR(20) NOT NULL,
                    order_code VARCHAR(20) NOT NULL,
                    guest_number INTEGER NOT NULL,
                    full_name VARCHAR(200) NOT NULL,
                    guest_hash VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scanned_at TIMESTAMP,
                    scanned_by VARCHAR(100),
                    scan_attempts INTEGER DEFAULT 0,
                    last_scan_attempt TIMESTAMP,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
                    UNIQUE(order_id, guest_number)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS promo_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(20) UNIQUE NOT NULL,
                    discount_type VARCHAR(10) DEFAULT 'percent',
                    discount_value INTEGER NOT NULL,
                    max_uses INTEGER DEFAULT 1,
                    used_count INTEGER DEFAULT 0,
                    valid_until TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR(100)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS action_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id BIGINT NOT NULL,
                    action_type VARCHAR(50) NOT NULL,
                    action_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scanner_id BIGINT NOT NULL,
                    scanner_username VARCHAR(100),
                    order_code VARCHAR(20) NOT NULL,
                    guest_name VARCHAR(200),
                    guest_hash VARCHAR(64),
                    scan_result VARCHAR(20),
                    scan_message TEXT,
                    scan_time_ms INTEGER,
                    qr_version VARCHAR(10),
                    signature_valid BOOLEAN,
                    timestamp_valid BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scanner_id BIGINT NOT NULL,
                    order_code VARCHAR(20) NOT NULL,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT FALSE,
                    UNIQUE(scanner_id, order_code, attempt_time)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qr_cache_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action VARCHAR(50),
                    cache_key VARCHAR(64),
                    cache_hit BOOLEAN,
                    generation_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_code ON orders(order_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_qr_hash ON orders(qr_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_guests_order_id ON guests(order_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_guests_order_code ON guests(order_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_guests_guest_hash ON guests(guest_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON bot_users(role)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_promo_codes_code ON promo_codes(code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_promo_codes_active ON promo_codes(is_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_user_id ON action_logs(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_created_at ON action_logs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_logs_scanner ON scan_logs(scanner_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_logs_code ON scan_logs(order_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_logs_created ON scan_logs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_attempts_scanner ON scan_attempts(scanner_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_attempts_code ON scan_attempts(order_code)")
            
            conn.commit()
            logger.info("✅ Таблицы SQLite базы данных инициализированы")
    
    def add_column_if_not_exists(self, table_name: str, column_name: str, column_type: str):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                if column_name not in column_names:
                    if "DEFAULT CURRENT_TIMESTAMP" in column_type.upper():
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} TIMESTAMP")
                        conn.commit()
                        
                        cursor.execute(f"UPDATE {table_name} SET {column_name} = CURRENT_TIMESTAMP WHERE {column_name} IS NULL")
                        conn.commit()
                    else:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                        conn.commit()
                    
                    logger.info(f"✅ Добавлена колонка {column_name} в таблицу {table_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка добавления колонки {column_name}: {e}")
            return False
    
    def check_and_fix_database(self):
        logger.info("🔧 Проверка структуры базы данных...")
        
        self.add_column_if_not_exists("orders", "ticket_type", "VARCHAR(10) DEFAULT 'standard'")
        self.add_column_if_not_exists("bot_users", "notified_about_restart", "BOOLEAN DEFAULT FALSE")
        self.add_column_if_not_exists("orders", "notified_promoters", "BOOLEAN DEFAULT FALSE")
        self.add_column_if_not_exists("bot_users", "request_count", "INTEGER DEFAULT 0")
        self.add_column_if_not_exists("bot_users", "last_request", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self.add_column_if_not_exists("orders", "processed_at", "TIMESTAMP")
        self.add_column_if_not_exists("orders", "scanned_at", "TIMESTAMP")
        self.add_column_if_not_exists("orders", "scanned_by", "VARCHAR(100)")
        self.add_column_if_not_exists("orders", "qr_hash", "VARCHAR(64)")
        self.add_column_if_not_exists("orders", "qr_version", "VARCHAR(10)")
        self.add_column_if_not_exists("guests", "scanned_at", "TIMESTAMP")
        self.add_column_if_not_exists("guests", "scanned_by", "VARCHAR(100)")
        self.add_column_if_not_exists("guests", "guest_hash", "VARCHAR(64)")
        self.add_column_if_not_exists("guests", "scan_attempts", "INTEGER DEFAULT 0")
        self.add_column_if_not_exists("guests", "last_scan_attempt", "TIMESTAMP")
        self.add_column_if_not_exists("scan_logs", "guest_hash", "VARCHAR(64)")
        self.add_column_if_not_exists("scan_logs", "scan_time_ms", "INTEGER")
        self.add_column_if_not_exists("scan_logs", "qr_version", "VARCHAR(10)")
        self.add_column_if_not_exists("scan_logs", "signature_valid", "BOOLEAN")
        self.add_column_if_not_exists("scan_logs", "timestamp_valid", "BOOLEAN")
        
        logger.info("✅ Структура базы данных проверена")
    
    def add_user(self, user_id: int, username: str = None, first_name: str = None, last_name: str = None):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                role = self._get_user_role(user_id)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO bot_users 
                    (user_id, username, first_name, last_name, role, last_active, is_active, request_count)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, TRUE, 
                    COALESCE((SELECT request_count FROM bot_users WHERE user_id = ?), 0) + 1)
                """, (user_id, username, first_name, last_name, role, user_id))
                
                conn.commit()
                logger.info(f"✅ Пользователь {user_id} добавлен/обновлен")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления пользователя {user_id}: {e}")
            return False
    
    def update_user_request(self, user_id: int):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE bot_users 
                    SET request_count = request_count + 1, 
                        last_request = CURRENT_TIMESTAMP,
                        last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления запросов пользователя {user_id}: {e}")
            return False
    
    def mark_user_notified(self, user_id: int):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE bot_users 
                    SET notified_about_restart = TRUE 
                    WHERE user_id = ?
                """, (user_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статуса уведомления для пользователя {user_id}: {e}")
            return False
    
    def reset_notification_status(self):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE bot_users SET notified_about_restart = FALSE")
                conn.commit()
                logger.info("✅ Статус уведомлений сброшен для всех пользователей")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка сброса статуса уведомлений: {e}")
            return False
    
    def get_users_to_notify(self) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM bot_users 
                    WHERE is_active = TRUE 
                    AND notified_about_restart = FALSE
                """)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения пользователей для уведомления: {e}")
            return []
    
    def _get_user_role(self, user_id: int) -> str:
        if user_id in ADMIN_IDS:
            return "admin"
        elif user_id in PROMOTER_IDS:
            return "promoter"
        else:
            return "user"
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM bot_users WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Ошибка получения пользователя {user_id}: {e}")
            return None
    
    def get_all_users(self) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM bot_users WHERE is_active = TRUE")
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения всех пользователей: {e}")
            return []
    
    def get_promoters(self) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM bot_users WHERE role = 'promoter' AND is_active = TRUE")
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения промоутеров: {e}")
            return []
    
    def get_scanners(self) -> List[Dict]:
        try:
            scanners = []
            for admin_id in ADMIN_IDS:
                user = self.get_user(admin_id)
                if user:
                    scanners.append(user)
            
            for promoter_id in PROMOTER_IDS:
                if promoter_id not in ADMIN_IDS:
                    user = self.get_user(promoter_id)
                    if user:
                        scanners.append(user)
            
            return scanners
        except Exception as e:
            logger.error(f"❌ Ошибка получения сканеров: {e}")
            return []
    
    def get_top_users(self, limit: int = 10) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, username, first_name, last_name, request_count, last_active
                    FROM bot_users 
                    WHERE is_active = TRUE 
                    ORDER BY request_count DESC 
                    LIMIT ?
                """, (limit,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения топ пользователей: {e}")
            return []
    
    def create_order(self, user_id: int, username: str, user_name: str, 
                    user_email: str, group_size: int, ticket_type: str, total_amount: int) -> Dict:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COALESCE(MAX(CAST(SUBSTR(order_id, 3) AS INTEGER)), 999) FROM orders")
                max_id = cursor.fetchone()[0] or 999
                order_id = f"SP{max_id + 1}"
                
                order_code = generate_unique_code()
                while self.get_order_by_code(order_code):
                    order_code = generate_unique_code()
                
                cursor.execute("""
                    INSERT INTO orders 
                    (order_id, order_code, user_id, username, user_name, user_email, 
                     group_size, ticket_type, total_amount, status, notified_promoters)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', FALSE)
                """, (order_id, order_code, user_id, username, user_name, user_email, 
                      group_size, ticket_type, total_amount))
                
                conn.commit()
                logger.info(f"✅ Заказ {order_id} создан, код: {order_code}, тип: {ticket_type}")
                log_user_action(user_id, "create_order", f"order_id={order_id}")
                
                return {
                    'order_id': order_id,
                    'order_code': order_code,
                    'user_id': user_id,
                    'username': username,
                    'user_name': user_name,
                    'user_email': user_email,
                    'group_size': group_size,
                    'ticket_type': ticket_type,
                    'total_amount': total_amount,
                    'status': 'active'
                }
        except Exception as e:
            logger.error(f"❌ Ошибка создания заказа: {e}")
            return None
    
    def mark_order_notified(self, order_id: str):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE orders 
                    SET notified_promoters = TRUE 
                    WHERE order_id = ?
                """, (order_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статуса уведомления для заказа {order_id}: {e}")
            return False
    
    def mark_order_processed(self, order_id: str):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE orders 
                    SET processed_at = CURRENT_TIMESTAMP 
                    WHERE order_id = ?
                """, (order_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления времени обработки заказа {order_id}: {e}")
            return False
    
    def get_unnotified_orders(self) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM orders 
                    WHERE status = 'active' 
                    AND notified_promoters = FALSE
                    ORDER BY created_at
                """)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения неуведомленных заказов: {e}")
            return []
    
    def get_old_unprocessed_orders(self, hours: int = 1) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM orders 
                    WHERE status = 'active' 
                    AND notified_promoters = TRUE
                    AND datetime(created_at) <= datetime('now', ?)
                    ORDER BY created_at
                """, (f'-{hours} hours',))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения старых заказов: {e}")
            return []
    
    def add_guests_to_order(self, order_id: str, order_code: str, guests: List[str]):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                for i, guest_name in enumerate(guests, 1):
                    guest_hash = hashlib.md5(guest_name.encode()).hexdigest()[:8] if guest_name else None
                    
                    cursor.execute("""
                        INSERT INTO guests (order_id, order_code, guest_number, full_name, guest_hash)
                        VALUES (?, ?, ?, ?, ?)
                    """, (order_id, order_code, i, guest_name.strip(), guest_hash))
                
                conn.commit()
                logger.info(f"✅ Добавлено {len(guests)} гостей к заказу {order_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления гостей к заказу {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Ошибка получения заказа {order_id}: {e}")
            return None
    
    def get_order_by_code(self, order_code: str) -> Optional[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM orders WHERE order_code = ?", (order_code,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Ошибка получения заказа по коду {order_code}: {e}")
            return None
    
    def get_user_orders(self, user_id: int) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения заказов пользователя {user_id}: {e}")
            return []
    
    def get_orders_by_status(self, status: str) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM orders WHERE status = ? ORDER BY created_at", (status,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения заказов со статусом {status}: {e}")
            return []
    
    def update_order_status(self, order_id: str, status: str, promoter_username: str = None) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                if status in ["closed", "refunded"]:
                    cursor.execute("""
                        UPDATE orders 
                        SET status = ?, closed_by = ?, closed_at = CURRENT_TIMESTAMP
                        WHERE order_id = ?
                    """, (status, promoter_username, order_id))
                elif status in ["active", "deferred"]:
                    cursor.execute("""
                        UPDATE orders 
                        SET status = ?, assigned_promoter = ?
                        WHERE order_id = ?
                    """, (status, promoter_username, order_id))
                else:
                    cursor.execute("""
                        UPDATE orders 
                        SET status = ?
                        WHERE order_id = ?
                    """, (status, order_id))
                
                conn.commit()
                logger.info(f"✅ Статус заказа {order_id} изменен на {status}")
                log_user_action(promoter_username or "system", "update_order_status", f"order_id={order_id}, status={status}")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статуса заказа {order_id}: {e}")
            return False
    
    def mark_ticket_scanned(self, order_code: str, scanner_id: int, scanner_username: str, guest_name: str = None) -> bool:
        log_details = {
            "order_code": order_code,
            "scanner_id": scanner_id,
            "scanner_username": scanner_username,
            "guest_name": guest_name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            logger.info(f"🔍 Попытка отметить билет {order_code} как отсканированный пользователем {scanner_id}")
            
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT scanned_at, scanned_by FROM orders WHERE order_code = ?", (order_code,))
                result = cursor.fetchone()
                
                if result and result[0] is not None:
                    logger.warning(f"⚠️ Билет {order_code} уже был отсканирован {result[0]} пользователем {result[1]}")
                    log_details["already_scanned"] = {
                        "scanned_at": str(result[0]),
                        "scanned_by": result[1]
                    }
                    log_details["success"] = False
                    log_qr_action("scan_already_used", log_details)
                    
                    if guest_name:
                        cursor.execute("""
                            UPDATE guests 
                            SET scan_attempts = scan_attempts + 1,
                                last_scan_attempt = CURRENT_TIMESTAMP
                            WHERE order_code = ? AND full_name = ?
                        """, (order_code, guest_name))
                        conn.commit()
                    
                    return False
                
                cursor.execute("""
                    UPDATE orders 
                    SET scanned_at = CURRENT_TIMESTAMP, 
                        scanned_by = ?
                    WHERE order_code = ? AND scanned_at IS NULL
                """, (scanner_username, order_code))
                
                order_updated = cursor.rowcount > 0
                log_details["order_updated"] = order_updated
                
                if guest_name:
                    cursor.execute("""
                        UPDATE guests 
                        SET scanned_at = CURRENT_TIMESTAMP, 
                            scanned_by = ?,
                            scan_attempts = scan_attempts + 1,
                            last_scan_attempt = CURRENT_TIMESTAMP
                        WHERE order_code = ? AND full_name = ? AND scanned_at IS NULL
                    """, (scanner_username, order_code, guest_name))
                    
                    guest_updated = cursor.rowcount > 0
                    log_details["guest_updated"] = guest_updated
                else:
                    cursor.execute("""
                        UPDATE guests 
                        SET scanned_at = CURRENT_TIMESTAMP, 
                            scanned_by = ?,
                            scan_attempts = scan_attempts + 1,
                            last_scan_attempt = CURRENT_TIMESTAMP
                        WHERE order_code = ? AND scanned_at IS NULL
                    """, (scanner_username, order_code))
                    
                    guests_updated = cursor.rowcount
                    log_details["guests_updated"] = guests_updated
                
                conn.commit()
                
                success = order_updated
                log_details["success"] = success
                
                if success:
                    logger.info(f"✅ Билет {order_code} успешно отмечен как использованный")
                    log_qr_action("scan_success", log_details)
                else:
                    logger.warning(f"⚠️ Не удалось отметить билет {order_code} как использованный")
                    log_qr_action("scan_failed", log_details)
                
                return success
                
        except Exception as e:
            log_details["error"] = str(e)
            log_details["traceback"] = traceback.format_exc()
            log_details["success"] = False
            logger.error(f"❌ Ошибка отметки билета как использованного: {e}")
            logger.error(f"📝 Traceback: {traceback.format_exc()}")
            log_qr_action("scan_error", log_details)
            return False
    
    def update_order_qr_data(self, order_id: str, qr_hash: str, qr_version: str) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE orders 
                    SET qr_hash = ?, qr_version = ?
                    WHERE order_id = ?
                """, (qr_hash, qr_version, order_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления QR данных: {e}")
            return False
    
    def update_guest_hash(self, order_code: str, guest_name: str, guest_hash: str) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE guests 
                    SET guest_hash = ?
                    WHERE order_code = ? AND full_name = ?
                """, (guest_hash, order_code, guest_name))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления хэша гостя: {e}")
            return False
    
    def log_scan(self, scanner_id: int, scanner_username: str, order_code: str, 
                 guest_name: str, result: str, message: str, scan_time_ms: int = None,
                 guest_hash: str = None, qr_version: str = None,
                 signature_valid: bool = None, timestamp_valid: bool = None):
        log_details = {
            "scanner_id": scanner_id,
            "scanner_username": scanner_username,
            "order_code": order_code,
            "guest_name": guest_name,
            "result": result,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO scan_logs 
                    (scanner_id, scanner_username, order_code, guest_name, guest_hash, 
                     scan_result, scan_message, scan_time_ms, qr_version, 
                     signature_valid, timestamp_valid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (scanner_id, scanner_username, order_code, guest_name, guest_hash,
                      result, message, scan_time_ms, qr_version,
                      signature_valid, timestamp_valid))
                conn.commit()
                
            logger.info(f"📝 Лог сканирования сохранен: {scanner_username} - {order_code} - {result}")
            log_qr_action("scan_logged", log_details)
            return True
        except Exception as e:
            log_details["db_error"] = str(e)
            logger.error(f"❌ Ошибка логирования сканирования: {e}")
            log_qr_action("scan_log_error", log_details)
            return False
    
    def record_scan_attempt(self, scanner_id: int, order_code: str, success: bool) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO scan_attempts (scanner_id, order_code, success)
                    VALUES (?, ?, ?)
                """, (scanner_id, order_code, success))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка записи попытки сканирования: {e}")
            return False
    
    def get_scan_attempts_count(self, scanner_id: int, minutes: int = 5) -> int:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM scan_attempts
                    WHERE scanner_id = ? 
                    AND attempt_time >= datetime('now', ?)
                """, (scanner_id, f'-{minutes} minutes'))
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"❌ Ошибка получения количества попыток: {e}")
            return 0
    
    def get_scan_stats(self) -> Dict:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM scan_logs")
                total_scans = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE scan_result = 'success'")
                success_scans = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE scan_result = 'error'")
                error_scans = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE scan_result = 'warning'")
                warning_scans = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT scanner_username, COUNT(*) as scan_count
                    FROM scan_logs
                    GROUP BY scanner_username
                    ORDER BY scan_count DESC
                    LIMIT 5
                """)
                top_scanners = cursor.fetchall()
                
                cursor.execute("""
                    SELECT COUNT(*) FROM orders 
                    WHERE scanned_at IS NOT NULL
                """)
                scanned_tickets = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'closed'")
                total_valid_tickets = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE DATE(created_at) = DATE('now')")
                today_scans = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE DATE(created_at) = DATE('now') AND scan_result = 'success'")
                today_success = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT 
                        strftime('%H', created_at) as hour,
                        COUNT(*) as scans
                    FROM scan_logs
                    WHERE created_at >= date('now', '-1 day')
                    GROUP BY hour
                    ORDER BY hour
                """)
                hourly_stats = cursor.fetchall()
                
                cursor.execute("""
                    SELECT 
                        scanner_username, 
                        order_code, 
                        scan_result, 
                        scan_time_ms,
                        signature_valid,
                        timestamp_valid,
                        created_at 
                    FROM scan_logs 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                recent_scans = cursor.fetchall()
                
                return {
                    "total_scans": total_scans,
                    "success_scans": success_scans,
                    "error_scans": error_scans,
                    "warning_scans": warning_scans,
                    "scanned_tickets": scanned_tickets,
                    "total_valid_tickets": total_valid_tickets,
                    "today_scans": today_scans,
                    "today_success": today_success,
                    "hourly_stats": [{"hour": h, "scans": s} for h, s in hourly_stats],
                    "top_scanners": [dict(row) for row in top_scanners],
                    "recent_scans": [
                        {
                            "scanner": s, 
                            "code": c, 
                            "result": r,
                            "time_ms": t,
                            "signature_valid": sv,
                            "timestamp_valid": tv,
                            "created_at": ca
                        } 
                        for s, c, r, t, sv, tv, ca in recent_scans
                    ]
                }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики сканирований: {e}")
            return {}
    
    def get_qr_statistics(self) -> Dict:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_scans,
                        SUM(CASE WHEN scan_result = 'success' THEN 1 ELSE 0 END) as success_scans,
                        SUM(CASE WHEN scan_result = 'warning' THEN 1 ELSE 0 END) as warning_scans,
                        SUM(CASE WHEN scan_result = 'error' THEN 1 ELSE 0 END) as error_scans,
                        AVG(scan_time_ms) as avg_scan_time
                    FROM scan_logs
                """)
                row = cursor.fetchone()
                
                cursor.execute("""
                    SELECT 
                        strftime('%H', created_at) as hour,
                        COUNT(*) as scans
                    FROM scan_logs
                    WHERE created_at >= date('now', '-1 day')
                    GROUP BY hour
                    ORDER BY hour
                """)
                hourly_stats = cursor.fetchall()
                
                cursor.execute("""
                    SELECT 
                        scanner_username,
                        COUNT(*) as scan_count,
                        SUM(CASE WHEN scan_result = 'success' THEN 1 ELSE 0 END) as success_count
                    FROM scan_logs
                    GROUP BY scanner_username
                    ORDER BY scan_count DESC
                    LIMIT 10
                """)
                top_scanners = cursor.fetchall()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_tickets,
                        SUM(CASE WHEN scanned_at IS NOT NULL THEN 1 ELSE 0 END) as scanned_tickets
                    FROM orders
                    WHERE status = 'closed'
                """)
                tickets_row = cursor.fetchone()
                
                cursor.execute("""
                    SELECT 
                        scanner_username,
                        order_code,
                        scan_result,
                        scan_time_ms,
                        signature_valid,
                        timestamp_valid,
                        created_at
                    FROM scan_logs 
                    ORDER BY created_at DESC 
                    LIMIT 20
                """)
                recent_scans = cursor.fetchall()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_cache_ops,
                        SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                        AVG(generation_time_ms) as avg_gen_time
                    FROM qr_cache_stats
                    WHERE created_at >= date('now', '-1 day')
                """)
                cache_row = cursor.fetchone()
                
                return {
                    "total_scans": row[0] or 0,
                    "success_scans": row[1] or 0,
                    "warning_scans": row[2] or 0,
                    "error_scans": row[3] or 0,
                    "avg_scan_time": round(row[4] or 0, 2),
                    "hourly_stats": [{"hour": h, "scans": s} for h, s in hourly_stats],
                    "top_scanners": [
                        {"username": u, "scans": c, "success": s} 
                        for u, c, s in top_scanners
                    ],
                    "total_tickets": tickets_row[0] or 0,
                    "scanned_tickets": tickets_row[1] or 0,
                    "recent_scans": [
                        {
                            "scanner": s, 
                            "code": c, 
                            "result": r,
                            "time_ms": t,
                            "signature_valid": sv,
                            "timestamp_valid": tv,
                            "created_at": ca
                        } 
                        for s, c, r, t, sv, tv, ca in recent_scans
                    ],
                    "cache_hits": cache_row[1] or 0,
                    "cache_misses": (cache_row[0] or 0) - (cache_row[1] or 0),
                    "cache_hit_rate": round(
                        ((cache_row[1] or 0) / max(cache_row[0] or 1, 1)) * 100, 2
                    ),
                    "avg_generation_time": round(cache_row[2] or 0, 2)
                }
        except Exception as e:
            logger.error(f"❌ Ошибка получения QR статистики: {e}")
            return {}
    
    def log_qr_cache(self, action: str, cache_key: str, cache_hit: bool, gen_time_ms: int):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO qr_cache_stats (action, cache_key, cache_hit, generation_time_ms)
                    VALUES (?, ?, ?, ?)
                """, (action, cache_key, cache_hit, gen_time_ms))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Ошибка логирования кэша: {e}")
    
    def get_recent_scan_attempts(self, scanner_id: int, limit: int = 10) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM scan_attempts
                    WHERE scanner_id = ?
                    ORDER BY attempt_time DESC
                    LIMIT ?
                """, (scanner_id, limit))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения попыток сканирования: {e}")
            return []
    
    def get_order_guests(self, order_id: str) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM guests WHERE order_id = ? ORDER BY guest_number", (order_id,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения гостей заказа {order_id}: {e}")
            return []
    
    def get_all_guests_count(self) -> int:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM guests")
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"❌ Ошибка получения общего количества гостей: {e}")
            return 0
    
    def reset_guests_count(self) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM guests")
                conn.commit()
                logger.info("✅ Счетчик гостей сброшен")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка сброса счетчика гостей: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT setting_value FROM event_settings WHERE setting_key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    try:
                        return json.loads(result[0])
                    except:
                        return result[0]
                return default
        except Exception as e:
            logger.error(f"❌ Ошибка получения настройки {key}: {e}")
            return default
    
    def set_setting(self, key: str, value: Any) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                if isinstance(value, (dict, list)):
                    value_json = json.dumps(value, ensure_ascii=False)
                else:
                    value_json = str(value)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO event_settings (setting_key, setting_value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, value_json))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка установки настройки {key}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM orders")
                total_orders = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'active'")
                active_orders = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'deferred'")
                deferred_orders = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'closed'")
                closed_orders = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'refunded'")
                refunded_orders = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE status = 'closed'")
                revenue = cursor.fetchone()[0] or 0
                
                total_guests = self.get_all_guests_count()
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE ticket_type = 'vip' AND status = 'closed'")
                vip_tickets = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE ticket_type = 'standard' AND status = 'closed'")
                standard_tickets = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE ticket_type = 'vip' AND status = 'closed'")
                vip_revenue = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE ticket_type = 'standard' AND status = 'closed'")
                standard_revenue = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM orders WHERE DATE(created_at) = DATE('now')")
                today_orders = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE DATE(created_at) = DATE('now') AND status = 'closed'")
                today_revenue = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM orders WHERE DATE(created_at) = DATE('now')")
                today_users = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as orders,
                        SUM(CASE WHEN status = 'closed' THEN total_amount ELSE 0 END) as revenue
                    FROM orders 
                    WHERE created_at >= DATE('now', '-7 days')
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """)
                weekly_stats = cursor.fetchall()
                
                weekly_data = []
                for row in weekly_stats:
                    weekly_data.append({
                        "date": row[0],
                        "orders": row[1] or 0,
                        "revenue": row[2] or 0
                    })
                
                cursor.execute("""
                    SELECT closed_by, COUNT(*) as closed_count, SUM(total_amount) as total_revenue
                    FROM orders 
                    WHERE status = 'closed' AND closed_by IS NOT NULL
                    GROUP BY closed_by
                    ORDER BY closed_count DESC
                    LIMIT 5
                """)
                top_promoters = cursor.fetchall()
                
                promoters_data = []
                for row in top_promoters:
                    promoters_data.append({
                        "username": row[0],
                        "closed_count": row[1] or 0,
                        "total_revenue": row[2] or 0
                    })
                
                return {
                    "total_orders": total_orders,
                    "active_orders": active_orders,
                    "deferred_orders": deferred_orders,
                    "closed_orders": closed_orders,
                    "refunded_orders": refunded_orders,
                    "revenue": revenue,
                    "total_guests": total_guests,
                    "vip_tickets": vip_tickets,
                    "standard_tickets": standard_tickets,
                    "vip_revenue": vip_revenue,
                    "standard_revenue": standard_revenue,
                    "today_orders": today_orders,
                    "today_revenue": today_revenue,
                    "today_users": today_users,
                    "weekly_stats": weekly_data,
                    "top_promoters": promoters_data
                }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def create_promo_code(self, code: str, discount_type: str, discount_value: int, 
                         max_uses: int = 1, valid_until: str = None, created_by: str = None) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO promo_codes 
                    (code, discount_type, discount_value, max_uses, valid_until, created_by, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, TRUE)
                """, (code, discount_type, discount_value, max_uses, valid_until, created_by))
                
                conn.commit()
                logger.info(f"✅ Промокод {code} создан")
                log_user_action(created_by or "system", "create_promo_code", f"code={code}")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка создания промокода: {e}")
            return False
    
    def get_promo_code(self, code: str) -> Optional[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM promo_codes WHERE code = ?", (code,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"❌ Ошибка получения промокода: {e}")
            return None
    
    def apply_promo_code(self, code: str, order_amount: int) -> Dict:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM promo_codes 
                    WHERE code = ? AND is_active = TRUE 
                    AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
                    AND (max_uses IS NULL OR used_count < max_uses)
                """, (code,))
                
                promo = cursor.fetchone()
                
                if not promo:
                    return {"success": False, "error": "Промокод не найден или недействителен"}
                
                promo_dict = dict(promo)
                
                discount = 0
                if promo_dict['discount_type'] == 'percent':
                    discount = order_amount * promo_dict['discount_value'] / 100
                else:
                    discount = min(promo_dict['discount_value'], order_amount)
                
                final_amount = order_amount - discount
                
                cursor.execute("""
                    UPDATE promo_codes 
                    SET used_count = used_count + 1 
                    WHERE id = ? AND (max_uses IS NULL OR used_count < max_uses)
                """, (promo_dict['id'],))
                
                conn.commit()
                
                if cursor.rowcount == 0:
                    return {"success": False, "error": "Лимит использования промокода исчерпан"}
                
                log_user_action("system", "apply_promo_code", f"code={code}, discount={discount}")
                
                return {
                    "success": True,
                    "discount": int(discount),
                    "final_amount": int(final_amount),
                    "promo_data": promo_dict
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка применения промокода: {e}")
            return {"success": False, "error": str(e)}
    
    def deactivate_promo_code(self, code: str) -> bool:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE promo_codes 
                    SET is_active = FALSE 
                    WHERE code = ?
                """, (code,))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Промокод {code} деактивирован")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка деактивации промокода: {e}")
            return False
    
    def get_all_promo_codes(self) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM promo_codes ORDER BY created_at DESC")
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения промокодов: {e}")
            return []
    
    def log_action(self, user_id: int, action_type: str, action_details: str = None):
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO action_logs (user_id, action_type, action_details)
                    VALUES (?, ?, ?)
                """, (user_id, action_type, action_details))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка записи действия в лог: {e}")
            return False
    
    def get_recent_actions(self, limit: int = 50) -> List[Dict]:
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT al.*, bu.username, bu.first_name, bu.last_name
                    FROM action_logs al
                    LEFT JOIN bot_users bu ON al.user_id = bu.user_id
                    ORDER BY al.created_at DESC
                    LIMIT ?
                """, (limit,))
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Ошибка получения действий: {e}")
            return []

class EventSettings:
    DEFAULT_SETTINGS = {
        "event_name": "SMILE PARTY 🎉",
        "event_date": "25 декабря 2024",
        "event_time": "20:00 - 06:00",
        "event_address": "Москва, ул. Праздничная, 17 (м. Радостная)",
        "event_age_limit": "18+",
        "contact_telegram": "@smile_party",
        "price_standard": 450,
        "price_group": 350,
        "price_vip": 650,
        "group_threshold": 5,
        "description": "Самое громкое мероприятие сезона! Топовые DJ-сеты, live-выступления, конкурсы с призами.",
        "event_info_text": "🏢 *Информация о мероприятии*\n\n*🎉 Название:* SMILE PARTY 🎉\n*📍 Адрес:* Москва, ул. Праздничная, 17 (м. Радостная)\n*📅 Дата:* 25 декабря 2024\n*⏰ Время:* 20:00 - 06:00\n*🎭 Возраст:* 18+\n*📱 Telegram:* @smile_party\n\n*📝 Описание:*\nСамое громкое мероприятие сезона! Топовые DJ-сеты, live-выступления, конкурсы с призами."
    }
    
    def __init__(self, db: Database):
        self.db = db
        self._load_defaults()
    
    def _load_defaults(self):
        for key, value in self.DEFAULT_SETTINGS.items():
            current = self.db.get_setting(key)
            if current is None:
                self.db.set_setting(key, value)
    
    def get_all_settings(self) -> Dict:
        settings = {}
        for key in self.DEFAULT_SETTINGS.keys():
            value = self.db.get_setting(key)
            if value is not None:
                settings[key] = value
            else:
                settings[key] = self.DEFAULT_SETTINGS[key]
        return settings
    
    def get_price_standard(self) -> int:
        return self.db.get_setting("price_standard", 450)
    
    def get_price_group(self) -> int:
        return self.db.get_setting("price_group", 350)
    
    def get_price_vip(self) -> int:
        return self.db.get_setting("price_vip", 650)
    
    def get_group_threshold(self) -> int:
        return self.db.get_setting("group_threshold", 5)
    
    def calculate_price(self, group_size: int, ticket_type: str = "standard") -> int:
        if ticket_type == "vip":
            return group_size * self.get_price_vip()
        elif group_size >= self.get_group_threshold():
            return group_size * self.get_price_group()
        else:
            return group_size * self.get_price_standard()
    
    def update_setting(self, key: str, value: Any) -> bool:
        if key in self.DEFAULT_SETTINGS:
            return self.db.set_setting(key, value)
        return False
    
    def reset_to_defaults(self) -> bool:
        success = True
        for key, value in self.DEFAULT_SETTINGS.items():
            if not self.db.set_setting(key, value):
                success = False
        return success

db = Database(DB_FILE)
db.check_and_fix_database()
event_settings = EventSettings(db)

(
    ROLE_SELECTION,
    MAIN_MENU,
    BUY_TICKET_TYPE,
    BUY_NAME,
    BUY_EMAIL,
    BUY_GUESTS,
    BUY_CONFIRM,
    ADMIN_MENU,
    PROMOTER_MENU,
    ADMIN_EDIT,
    ADMIN_EDIT_TEXT,
    PROMOTER_VIEW_ORDER,
    PROMOTER_DEFERRED,
    ADMIN_RESET_STATS,
    ADMIN_CREATE_PROMO,
    ADMIN_VIEW_PROMO,
    ADMIN_BROADCAST,
    ADMIN_DASHBOARD,
    ADMIN_EXPORT_DATA,
    SCAN_QR,
    SCAN_RESULT
) = range(21)

def safe_markdown_text(text: str) -> str:
    if not text:
        return ""
    
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    
    result = ''
    for char in text:
        if char in escape_chars:
            result += '\\' + char
        else:
            result += char
    
    return result

def escape_markdown(text: str) -> str:
    if not text:
        return ""
    
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    
    result = ''
    for char in text:
        if char in escape_chars:
            result += '\\' + char
        else:
            result += char
    
    return result

def get_user_role(user_id: int) -> str:
    if user_id in ADMIN_IDS:
        return "admin"
    elif user_id in PROMOTER_IDS:
        return "promoter"
    else:
        return "user"

def is_valid_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_own_order(order: Dict, user_id: int) -> bool:
    return order["user_id"] == user_id

async def send_channel_notification(context: ContextTypes.DEFAULT_TYPE, order: Dict, promoter_username: str, action: str):
    try:
        formatted_code = format_code_for_display(order['order_code'])
        
        if action == "closed":
            channel_id = CLOSED_ORDERS_CHANNEL_ID
            closed_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
            
            ticket_type_text = "VIP 🎩" if order.get('ticket_type') == 'vip' else "Обычный 🎟"
            
            text = (
                "✅ *Заявка успешно обработана*\n\n"
                f"*Уникальный код:* `{order['order_code']}`\n"
                f"*Тип билета:* {ticket_type_text}\n"
                f"*ID заявки:* #{order['order_id']}\n"
                f"*Закрыл заявку:* @{escape_markdown(promoter_username)}\n"
                f"*Контактное лицо:* {escape_markdown(str(order['user_name']))}\n"
                f"*Telegram:* @{escape_markdown(str(order['username'] or 'без username'))}\n"
                f"*Email:* {escape_markdown(str(order['user_email']))}\n"
                f"*Дата закрытия:* {closed_time}\n"
                f"*Количество гостей:* {order['group_size']}\n"
                f"*Сумма:* {order['total_amount']} ₽"
            )
        elif action == "refunded":
            channel_id = REFUND_ORDERS_CHANNEL_ID
            closed_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
            
            ticket_type_text = "VIP 🎩" if order.get('ticket_type') == 'vip' else "Обычный 🎟"
            
            text = (
                "❌ *Возврат заявки*\n\n"
                f"*Уникальный код:* `{order['order_code']}`\n"
                f"*Тип билета:* {ticket_type_text}\n"
                f"*ID заявки:* #{order['order_id']}\n"
                f"*Промоутер:* @{escape_markdown(promoter_username)}\n"
                f"*Контактное лицо:* {escape_markdown(str(order['user_name']))}\n"
                f"*Telegram:* @{escape_markdown(str(order['username'] or 'без username'))}\n"
                f"*Email:* {escape_markdown(str(order['user_email']))}\n"
                f"*Дата возврата:* {closed_time}\n"
                f"*Количество гостей:* {order['group_size']}\n"
                f"*Сумма:* {order['total_amount']} ₽"
            )
        else:
            return
        
        await context.bot.send_message(
            chat_id=channel_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Уведомление отправлено в канал для заказа #{order['order_id']}")
        
    except Exception as e:
        logger.error(f"Ошибка отправки уведомления в канал: {e}")

async def send_to_lists_channel(context: ContextTypes.DEFAULT_TYPE, order: Dict, promoter_username: str):
    try:
        guests = db.get_order_guests(order['order_id'])
        closed_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        if not guests:
            return
        
        for guest in guests:
            guest_name = guest['full_name']
            
            name_parts = guest_name.strip().split()
            if len(name_parts) >= 2:
                last_name = name_parts[0]
                first_name = ' '.join(name_parts[1:])
            else:
                last_name = ""
                first_name = guest_name
            
            formatted_code = format_code_for_display(order['order_code'])
            
            ticket_type_text = "VIP 🎩" if order.get('ticket_type') == 'vip' else "Обычный 🎟"
            
            text = (
                f"✅ *Добавлен в список:*\n\n"
                f"*Фамилия:* {escape_markdown(last_name)}\n"
                f"*Имя:* {escape_markdown(first_name)}\n"
                f"*Тип билета:* {ticket_type_text}\n"
                f"*Контакт:* {escape_markdown(str(order['user_name']))}\n"
                f"*Telegram:* @{escape_markdown(str(order['username'] or 'без username'))}\n"
                f"*Уникальный код:* `{order['order_code']}`\n"
                f"*Время закрытия:* {closed_time}\n"
                f"*Промоутер:* @{escape_markdown(promoter_username)}"
            )
            
            await context.bot.send_message(
                chat_id=LISTS_CHANNEL_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN
            )
            
            await asyncio.sleep(0.5)
        
        logger.info(f"Информация о {len(guests)} гостях отправлена в канал списков для заказа #{order['order_id']}")
        
    except Exception as e:
        logger.error(f"Ошибка отправки информации в канал списков: {e}")

async def send_new_order_notification(context: ContextTypes.DEFAULT_TYPE, order: Dict):
    try:
        guests = db.get_order_guests(order['order_id'])
        
        created_at = order['created_at']
        if isinstance(created_at, str):
            created_date = created_at[:16].replace('T', ' ')
        else:
            created_date = created_at.strftime('%d.%m.%Y %H:%M')
        
        user_name = escape_markdown(str(order['user_name']))
        username = order['username'] if order['username'] else 'без username'
        escaped_username = escape_markdown(username)
        user_email = escape_markdown(str(order['user_email']))
        
        formatted_code = format_code_for_display(order['order_code'])
        
        ticket_type_text = "VIP 🎩" if order.get('ticket_type') == 'vip' else "Обычный 🎟"
        
        text = (
            "🆕 *Новая заявка!*\n\n"
            f"*Уникальный код:* `{order['order_code']}`\n"
            f"*Тип билета:* {ticket_type_text}\n"
            f"*ID заявки:* `{order['order_id']}`\n"
            f"*Контактное лицо:* {user_name}\n"
            f"*Telegram:* @{escaped_username}\n"
            f"*Email:* {user_email}\n"
            f"*User ID:* `{order['user_id']}`\n"
            f"*Количество человек:* {order['group_size']}\n"
            f"*Сумма заказа:* {order['total_amount']} ₽\n"
            f"*Дата создания:* {created_date}\n"
        )
        
        if guests:
            text += f"\n*Список гостей:*"
            for guest in guests:
                guest_name = escape_markdown(str(guest['full_name']))
                text += f"\n• {guest_name}"
        
        text += f"\n\n*💬 Способы связи:*"
        
        if username and username != 'без username' and username != 'None':
            clean_username = username.lstrip('@')
            text += f"\n• Telegram: @{clean_username}"
            text += f"\n• Ссылка: https://t.me/{clean_username}"
        else:
            text += f"\n• User ID: {order['user_id']}"
            text += f"\n• Ссылка: tg://user?id={order['user_id']}"
        
        text += f"\n• Email: {user_email}"
        
        bot_username = context.bot.username
        bot_link = f"https://t.me/{bot_username}?start=order_{order['order_id']}"
        
        keyboard = [
            [InlineKeyboardButton("📋 Обработать заявку в боте", url=bot_link)],
            [InlineKeyboardButton("💬 Написать в диалог", url=f"tg://user?id={order['user_id']}")]
        ]
        
        try:
            await context.bot.send_message(
                chat_id=PROMOTERS_CHAT_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            logger.info(f"Уведомление о новом заказе {order['order_id']} отправлено в чат промоутеров")
            
            db.mark_order_notified(order['order_id'])
            
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления в чат промоутеров: {e}")
            
    except Exception as e:
        logger.error(f"Ошибка при формировании уведомления о новом заказе: {e}")

async def check_and_send_notifications(context: ContextTypes.DEFAULT_TYPE):
    try:
        unnotified_orders = db.get_unnotified_orders()
        
        for order in unnotified_orders:
            await send_new_order_notification(context, order)
            await asyncio.sleep(1)
            
        if unnotified_orders:
            logger.info(f"Отправлено уведомлений о {len(unnotified_orders)} новых заказах")
            
    except Exception as e:
        logger.error(f"Ошибка при проверке и отправке уведомлений: {e}")

async def send_reminders(context: ContextTypes.DEFAULT_TYPE):
    try:
        old_orders = db.get_old_unprocessed_orders(hours=1)
        
        if old_orders:
            reminder_text = "⏰ *НАПОМИНАНИЕ!*\n\n"
            reminder_text += f"Следующие заказы активны более 1 часа:\n\n"
            
            for order in old_orders[:5]:
                reminder_text += f"• Заказ #{order['order_id']} ({order['order_code']}) - {order['user_name']}\n"
            
            if len(old_orders) > 5:
                reminder_text += f"\n...и еще {len(old_orders) - 5} заказов\n"
            
            reminder_text += "\nПожалуйста, обработайте эти заказы как можно скорее!"
            
            try:
                await context.bot.send_message(
                    chat_id=PROMOTERS_CHAT_ID,
                    text=reminder_text,
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.info(f"Отправлено напоминание о {len(old_orders)} старых заказах")
            except Exception as e:
                logger.error(f"Ошибка отправки напоминания: {e}")
                
    except Exception as e:
        logger.error(f"Ошибка в send_reminders: {e}")

async def send_order_notification_to_user(context: ContextTypes.DEFAULT_TYPE, order: Dict, action: str, promoter_username: str):
    try:
        if order['user_id']:
            escaped_promoter = escape_markdown(promoter_username)
            escaped_user_name = escape_markdown(str(order['user_name']))
            formatted_code = format_code_for_display(order['order_code'])
            
            ticket_type_text = "VIP" if order.get('ticket_type') == 'vip' else "Обычный"
            
            if action == "closed":
                message = (
                    f"✅ *Ваш заказ #{order['order_id']} успешно обработан!*\n\n"
                    f"*Тип билета:* {ticket_type_text}\n"
                    f"*Ваш уникальный код:* `{order['order_code']}`\n\n"
                    f"Промоутер @{escaped_promoter} подтвердил вашу покупку.\n\n"
                    f"*Детали заказа:*\n"
                    f"• Контактное лицо: {escaped_user_name}\n"
                    f"• Количество гостей: {order['group_size']}\n"
                    f"• Сумма: {order['total_amount']} ₽\n\n"
                    f"*💾 Сохраните ваш код! Он потребуется при входе на мероприятие.*\n\n"
                    f"Спасибо за покупку! Ждем вас на мероприятии! 🎉"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🎫 Получить QR-код билета", callback_data=f"get_qr_{order['order_id']}")]
                ])
                
                await context.bot.send_message(
                    chat_id=order['user_id'],
                    text=message,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.MARKDOWN
                )
            elif action == "refunded":
                message = (
                    f"❌ *По вашему заказу #{order['order_id']} оформлен возврат*\n\n"
                    f"*Тип билета:* {ticket_type_text}\n"
                    f"*Код заказа:* `{order['order_code']}`\n\n"
                    f"Промоутер @{escaped_promoter} оформил возврат по вашему заказу.\n\n"
                    f"Если у вас есть вопросы, свяжитесь с поддержкой: {event_settings.get_all_settings()['contact_telegram']}"
                )
                
                await context.bot.send_message(
                    chat_id=order['user_id'],
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                return
            
            logger.info(f"Уведомление отправлено пользователю {order['user_id']}")
    except Exception as e:
        logger.error(f"Ошибка отправки уведомления пользователю: {e}")

async def generate_ticket_qr(update: Update, context: ContextTypes.DEFAULT_TYPE, order_code: str):
    start_time = time.time()
    
    log_details = {
        "order_code": order_code,
        "user_id": update.effective_user.id if update.effective_user else None,
        "action": "generate_ticket_qr",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"🚀 Начало генерации QR-кода для заказа {order_code}")
        
        order = db.get_order_by_code(order_code)
        
        if not order:
            error_msg = "Заказ не найден"
            logger.warning(f"⚠️ {error_msg}: {order_code}")
            
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    "❌ *Заказ не найден*",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await update.message.reply_text(
                    "❌ *Заказ не найден*",
                    parse_mode=ParseMode.MARKDOWN
                )
            return
        
        if order['status'] != 'closed':
            error_msg = f"Билет еще не активирован (статус: {order['status']})"
            logger.warning(f"⚠️ {error_msg}")
            
            error_text = f"❌ *Билет еще не активирован!*\n\nСтатус заказа: {order['status']}\nQR-код будет доступен после подтверждения покупки промоутером."
            
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    error_text,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await update.message.reply_text(
                    error_text,
                    parse_mode=ParseMode.MARKDOWN
                )
            return
        
        guests = db.get_order_guests(order['order_id'])
        
        qr_hash = hashlib.md5(f"{order_code}_{order.get('ticket_type', 'standard')}".encode()).hexdigest()
        db.update_order_qr_data(order['order_id'], qr_hash, QR_CONFIG["version"])
        
        if guests:
            logger.info(f"📋 Найдено {len(guests)} гостей для заказа {order_code}")
            
            for i, guest in enumerate(guests, 1):
                guest_name = guest['full_name']
                
                guest_hash = hashlib.md5(guest_name.encode()).hexdigest()[:8]
                db.update_guest_hash(order_code, guest_name, guest_hash)
                
                qr_bytes = qr_manager.generate_qr_image(
                    order_code,
                    order.get('ticket_type', 'standard'),
                    guest_name
                )
                
                cache_key = hashlib.md5(f"{order_code}_{guest_name}".encode()).hexdigest()
                db.log_qr_cache(
                    "generate",
                    cache_key,
                    False,
                    int((time.time() - start_time) * 1000)
                )
                
                caption = (
                    f"🎫 *Билет для {escape_markdown(guest_name)}*\n\n"
                    f"🔑 *Код:* `{order_code}`\n"
                    f"🎫 *Тип:* {'VIP' if order.get('ticket_type') == 'vip' else 'Обычный'}\n"
                    f"🔒 *Защита:* HMAC + Timestamp\n"
                    f"📱 *Версия:* {QR_CONFIG['version']}\n"
                    f"👤 *Контакт:* {escape_markdown(str(order['user_name']))}"
                )
                
                if update.callback_query:
                    await update.callback_query.message.reply_photo(
                        photo=io.BytesIO(qr_bytes),
                        caption=caption,
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await update.message.reply_photo(
                        photo=io.BytesIO(qr_bytes),
                        caption=caption,
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                await asyncio.sleep(0.5)
        else:
            qr_bytes = qr_manager.generate_qr_image(
                order_code,
                order.get('ticket_type', 'standard')
            )
            
            caption = (
                f"🎫 *Билет для {escape_markdown(str(order['user_name']))}*\n\n"
                f"🔑 *Код:* `{order_code}`\n"
                f"🎫 *Тип:* {'VIP' if order.get('ticket_type') == 'vip' else 'Обычный'}\n"
                f"🔒 *Защита:* HMAC + Timestamp\n"
                f"📱 *Версия:* {QR_CONFIG['version']}"
            )
            
            if update.callback_query:
                await update.callback_query.message.reply_photo(
                    photo=io.BytesIO(qr_bytes),
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await update.message.reply_photo(
                    photo=io.BytesIO(qr_bytes),
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN
                )
        
        logger.info(f"✅ QR-коды успешно отправлены для заказа {order_code} за {time.time()-start_time:.2f}с")
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации QR-кода: {e}")
        logger.error(traceback.format_exc())
        
        error_text = f"❌ *Ошибка генерации QR-кода:*\n\n{str(e)}"
        
        if update.callback_query:
            await update.callback_query.message.reply_text(
                error_text,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                error_text,
                parse_mode=ParseMode.MARKDOWN
            )

async def scan_qr_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    logger.info(f"📱 Пользователь {user.id} (@{user.username}) вызвал команду scan_qr")
    
    if user.id not in SCANNER_IDS:
        logger.warning(f"⚠️ Пользователь {user.id} попытался сканировать QR без прав")
        await update.message.reply_text(
            "❌ *У вас нет прав для сканирования QR-кодов*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    await update.message.reply_text(
        "📱 *Сканирование QR-кода*\n\n"
        "Пожалуйста, отправьте фото QR-кода билета для проверки.\n\n"
        "Или введите код билета вручную (например: #KA123456)",
        parse_mode=ParseMode.MARKDOWN
    )
    
    context.user_data['scan_mode'] = True
    logger.info(f"✅ Режим сканирования активирован для пользователя {user.id}")
    return SCAN_QR

async def handle_qr_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    username = user.username or f"user_{user.id}"
    start_time = time.time()
    
    scan_log_details = {
        "scanner_id": user.id,
        "scanner_username": username,
        "scan_method": "text",
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"📱 Начало обработки сканирования QR от пользователя {user.id}")
    
    if user.id not in SCANNER_IDS:
        logger.warning(f"⚠️ Пользователь {user.id} попытался сканировать QR без прав")
        await update.message.reply_text(
            "❌ *У вас нет прав для сканирования QR-кодов*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    attempts = db.get_scan_attempts_count(user.id, 5)
    if attempts > 20:
        logger.warning(f"⚠️ Пользователь {user.id} превысил лимит попыток сканирования")
        await update.message.reply_text(
            "⏰ *Слишком много попыток сканирования!*\n\n"
            "Пожалуйста, подождите 5 минут перед повторной попыткой.",
            parse_mode=ParseMode.MARKDOWN
        )
        return SCAN_QR
    
    try:
        qr_data = None
        scan_result = None
        
        if update.message.photo:
            scan_log_details["scan_method"] = "photo"
            logger.info("📸 Получено фото для распознавания QR-кода")
            
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                await file.download_to_drive(tmp_file.name)
                tmp_path = tmp_file.name
            
            logger.info(f"📥 Фото сохранено во временный файл: {tmp_path}")
            
            try:
                with open(tmp_path, 'rb') as f:
                    image_bytes = f.read()
                
                scan_result = qr_manager.scan_qr_image(image_bytes)
                
                if scan_result["success"]:
                    qr_data = scan_result["data"]
                    logger.info(f"✅ QR-код распознан: {qr_data[:50]}...")
                else:
                    logger.warning(f"⚠️ QR-код не распознан: {scan_result['error']}")
            
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.error(f"❌ Ошибка удаления временного файла: {e}")
        
        elif update.message.text:
            text = update.message.text.strip()
            scan_log_details["input_text"] = text
            logger.info(f"📝 Получен текст для обработки: {text}")
            
            code_patterns = [
                r'#?KA\d{6}',
                r'KA\d{6}',
                r'\d{6}',
                r'SMILE_PARTY:.*'
            ]
            
            for pattern in code_patterns:
                match = re.search(pattern, text)
                if match:
                    qr_data = match.group()
                    scan_log_details["extracted_data"] = qr_data
                    logger.info(f"🔍 Извлечено из текста по шаблону {pattern}: {qr_data}")
                    break
            
            if not qr_data:
                scan_result = {"success": True, "data": text}
                qr_data = text
        
        if qr_data:
            if ':' in qr_data:
                parsed = qr_manager.parse_qr_data(qr_data)
                scan_log_details["parsed"] = parsed
                
                if not parsed["valid"]:
                    error_msg = parsed.get("error", "Неизвестная ошибка")
                    logger.warning(f"⚠️ QR-код не валиден: {error_msg}")
                    
                    result_text = f"❌ *{error_msg}*"
                    
                    db.log_scan(
                        user.id, username, 
                        parsed.get("code", "unknown"), 
                        None, "error", error_msg,
                        scan_time_ms=int((time.time() - start_time) * 1000),
                        qr_version=parsed.get("version"),
                        signature_valid=False,
                        timestamp_valid=False
                    )
                    
                    db.record_scan_attempt(user.id, "unknown", False)
                    
                    keyboard = [
                        [InlineKeyboardButton("📱 Сканировать еще", callback_data="scan_qr_start")],
                        [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_menu")]
                    ]
                    
                    await update.message.reply_text(
                        result_text,
                        reply_markup=InlineKeyboardMarkup(keyboard),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                    return SCAN_RESULT
                
                code = parsed["code"]
                ticket_type = parsed.get("ticket_type", "standard")
                guest_hash = parsed.get("guest_hash")
            else:
                code = qr_data.replace('#', '').strip()
                parsed = {"valid": True, "code": code}
            
            rate_ok, wait_time = qr_manager.check_scan_rate_limit(user.id, code)
            if not rate_ok:
                logger.warning(f"⚠️ Rate limit для билета {code}, сканер {user.id}")
                
                result_text = (
                    f"⏰ *Слишком частые попытки сканирования!*\n\n"
                    f"🔑 Код: `{code}`\n"
                    f"Пожалуйста, подождите {wait_time} секунд перед повторным сканированием."
                )
                
                db.log_scan(
                    user.id, username, code, None, "warning",
                    f"Rate limit, ожидание {wait_time}с",
                    scan_time_ms=int((time.time() - start_time) * 1000)
                )
                
                keyboard = [
                    [InlineKeyboardButton("📱 Сканировать еще", callback_data="scan_qr_start")],
                    [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_menu")]
                ]
                
                await update.message.reply_text(
                    result_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
                
                return SCAN_RESULT
            
            order = db.get_order_by_code(code)
            
            if not order:
                logger.warning(f"⚠️ Билет с кодом {code} не найден")
                result_text = "❌ *Билет не найден в системе!*"
                
                db.log_scan(
                    user.id, username, code, None, "error", 
                    "Билет не найден",
                    scan_time_ms=int((time.time() - start_time) * 1000)
                )
                db.record_scan_attempt(user.id, code, False)
                
            elif order['status'] != 'closed':
                logger.warning(f"⚠️ Билет {code} не активирован (статус: {order['status']})")
                result_text = f"❌ *Билет еще не активирован!*\n\nСтатус: {order['status']}"
                
                db.log_scan(
                    user.id, username, code, None, "error", 
                    f"Статус: {order['status']}",
                    scan_time_ms=int((time.time() - start_time) * 1000)
                )
                db.record_scan_attempt(user.id, code, False)
                
            elif order.get('scanned_at'):
                scanned_time = order['scanned_at']
                if isinstance(scanned_time, str):
                    scanned_time = scanned_time[:16]
                else:
                    scanned_time = scanned_time.strftime('%d.%m.%Y %H:%M')
                
                logger.warning(f"⚠️ Билет {code} уже был использован {scanned_time}")
                
                if order.get('scanned_by') == username:
                    result_text = (
                        f"⚠️ *Билет уже был отсканирован ВАМИ!*\n\n"
                        f"🔑 Код: `{code}`\n"
                        f"👤 Владелец: {escape_markdown(str(order['user_name']))}\n"
                        f"📅 Время сканирования: {scanned_time}\n\n"
                        f"❌ *Повторный вход запрещен!*"
                    )
                else:
                    result_text = (
                        f"⚠️ *Билет уже был использован!*\n\n"
                        f"🔑 Код: `{code}`\n"
                        f"👤 Владелец: {escape_markdown(str(order['user_name']))}\n"
                        f"📅 Время сканирования: {scanned_time}\n"
                        f"👨‍💼 Сканировал: @{order.get('scanned_by', 'неизвестно')}\n\n"
                        f"❌ *Повторный вход запрещен!*"
                    )
                
                db.log_scan(
                    user.id, username, code, None, "warning", 
                    "Повторное сканирование",
                    scan_time_ms=int((time.time() - start_time) * 1000),
                    signature_valid=parsed.get("valid", True) if 'parsed' in locals() else None,
                    timestamp_valid=parsed.get("valid", True) if 'parsed' in locals() else None
                )
                db.record_scan_attempt(user.id, code, False)
                
            else:
                logger.info(f"✅ Билет {code} найден и готов к сканированию")
                
                guest_match = True
                guest_name = None
                
                if guest_hash:
                    guests = db.get_order_guests(order['order_id'])
                    for guest in guests:
                        if guest.get('guest_hash') == guest_hash:
                            guest_name = guest['full_name']
                            break
                    
                    if not guest_name:
                        logger.warning(f"⚠️ Хэш гостя {guest_hash} не совпадает ни с одним гостем")
                        guest_match = False
                
                if not guest_match:
                    result_text = (
                        f"⚠️ *Несовпадение данных гостя!*\n\n"
                        f"🔑 Код: `{code}`\n"
                        f"Пожалуйста, попросите гостя показать свой личный QR-код."
                    )
                    
                    db.log_scan(
                        user.id, username, code, None, "error",
                        "Несовпадение хэша гостя",
                        scan_time_ms=int((time.time() - start_time) * 1000)
                    )
                    db.record_scan_attempt(user.id, code, False)
                else:
                    success = db.mark_ticket_scanned(code, user.id, username, guest_name)
                    
                    if success:
                        logger.info(f"✅ Билет {code} успешно отмечен как использованный")
                        
                        guests = db.get_order_guests(order['order_id'])
                        
                        if guests:
                            guest_list = "\n".join([f"• {escape_markdown(g['full_name'])}" for g in guests])
                            
                            scanned_guest_marker = ""
                            if guest_name:
                                scanned_guest_marker = f"\n✅ Отсканирован гость: {escape_markdown(guest_name)}"
                            
                            result_text = (
                                f"✅ *Билет действителен!*\n\n"
                                f"🔑 Код: `{code}`\n"
                                f"🎫 Тип: {'VIP 🎩' if order.get('ticket_type') == 'vip' else 'Обычный 🎟'}\n"
                                f"👤 Контакт: {escape_markdown(str(order['user_name']))}\n"
                                f"👥 Гости:\n{guest_list}"
                                f"{scanned_guest_marker}\n\n"
                                f"✅ *Вход разрешен!*\n\n"
                                f"📝 *Билет отмечен как использованный*"
                            )
                        else:
                            result_text = (
                                f"✅ *Билет действителен!*\n\n"
                                f"🔑 Код: `{code}`\n"
                                f"🎫 Тип: {'VIP 🎩' if order.get('ticket_type') == 'vip' else 'Обычный 🎟'}\n"
                                f"👤 Владелец: {escape_markdown(str(order['user_name']))}\n\n"
                                f"✅ *Вход разрешен!*\n\n"
                                f"📝 *Билет отмечен как использованный*"
                            )
                        
                        db.log_scan(
                            user.id, username, code, guest_name, "success",
                            "Успешное сканирование",
                            scan_time_ms=int((time.time() - start_time) * 1000),
                            guest_hash=guest_hash,
                            signature_valid=parsed.get("valid", True) if 'parsed' in locals() else None,
                            timestamp_valid=parsed.get("valid", True) if 'parsed' in locals() else None,
                            qr_version=parsed.get("version") if 'parsed' in locals() else None
                        )
                        db.record_scan_attempt(user.id, code, True)
                        
                        await send_log_to_channel(
                            context,
                            f"✅ QR-код отсканирован: {code} - гость: {guest_name or 'не указан'} - сканер: @{username}",
                            "INFO"
                        )
                    else:
                        logger.warning(f"⚠️ Ошибка при отметке билета {code}")
                        result_text = (
                            f"⚠️ *Ошибка при отметке билета*\n\n"
                            f"🔑 Код: `{code}`\n\n"
                            f"Возможно, билет уже был использован. Пожалуйста, проверьте вручную."
                        )
                        
                        db.log_scan(
                            user.id, username, code, guest_name, "error",
                            "Ошибка отметки билета",
                            scan_time_ms=int((time.time() - start_time) * 1000)
                        )
                        db.record_scan_attempt(user.id, code, False)
            
            keyboard = [
                [InlineKeyboardButton("📱 Сканировать еще", callback_data="scan_qr_start")],
                [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_menu")]
            ]
            
            await update.message.reply_text(
                result_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
            
            scan_log_details["success"] = True
            scan_log_details["scan_time_ms"] = int((time.time() - start_time) * 1000)
            
            return SCAN_RESULT
        
        else:
            logger.warning(f"⚠️ Не удалось распознать код")
            await update.message.reply_text(
                "❌ *Не удалось распознать QR-код*\n\n"
                "Пожалуйста, убедитесь что:\n"
                "• Фото четкое и хорошо освещено\n"
                "• QR-код занимает большую часть кадра\n"
                "• Нет бликов и искажений\n\n"
                "Или введите код вручную в формате #KA123456",
                parse_mode=ParseMode.MARKDOWN
            )
            return SCAN_QR
    
    except Exception as e:
        scan_log_details["error"] = str(e)
        scan_log_details["traceback"] = traceback.format_exc()
        logger.error(f"❌ Ошибка при сканировании QR-кода: {e}")
        logger.error(traceback.format_exc())
        
        await update.message.reply_text(
            f"❌ *Ошибка при обработке:*\n\n{str(e)}",
            parse_mode=ParseMode.MARKDOWN
        )
        return SCAN_QR

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await handle_qr_scan(update, context)

async def scan_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS + SCANNER_IDS:
        await update.message.reply_text(
            "❌ *У вас нет прав для просмотра статистики сканирований*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    stats = db.get_scan_stats()
    
    text = "📊 *Статистика сканирований QR-кодов*\n\n"
    text += f"• Всего сканирований: {stats.get('total_scans', 0)}\n"
    text += f"• ✅ Успешных: {stats.get('success_scans', 0)}\n"
    text += f"• ⚠️ Повторных: {stats.get('warning_scans', 0)}\n"
    text += f"• ❌ Ошибок: {stats.get('error_scans', 0)}\n"
    text += f"• 📱 Отсканировано билетов: {stats.get('scanned_tickets', 0)}/{stats.get('total_valid_tickets', 0)}\n"
    text += f"• 📅 Сегодня: {stats.get('today_scans', 0)} (успешно: {stats.get('today_success', 0)})\n\n"
    
    if stats.get('top_scanners'):
        text += "🏆 *Топ сканеров:*\n"
        for i, scanner in enumerate(stats['top_scanners'][:5], 1):
            text += f"{i}. @{scanner['scanner_username']}: {scanner['scan_count']} сканирований\n"
        text += "\n"
    
    if stats.get('recent_scans'):
        text += "📋 *Последние 5 сканирований:*\n"
        for scan in stats['recent_scans'][:5]:
            created_at = scan['created_at']
            if isinstance(created_at, str):
                time_str = created_at[11:16]
            else:
                time_str = created_at.strftime('%H:%M')
            
            emoji = "✅" if scan['scan_result'] == 'success' else "⚠️" if scan['scan_result'] == 'warning' else "❌"
            text += f"{emoji} {time_str} - @{scan['scanner_username']} - {scan['order_code']}\n"
    
    await update.message.reply_text(
        text,
        parse_mode=ParseMode.MARKDOWN
    )

async def qr_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS:
        await update.message.reply_text(
            "❌ *У вас нет прав для просмотра статистики*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    stats = db.get_qr_statistics()
    qr_manager_stats = qr_manager.get_stats()
    
    text = "📊 *РАСШИРЕННАЯ СТАТИСТИКА QR-КОДОВ*\n\n"
    
    text += "📈 *ОБЩАЯ СТАТИСТИКА:*\n"
    text += f"• Всего сканирований: {stats.get('total_scans', 0)}\n"
    text += f"• ✅ Успешных: {stats.get('success_scans', 0)}\n"
    text += f"• ⚠️ Повторных: {stats.get('warning_scans', 0)}\n"
    text += f"• ❌ Ошибок: {stats.get('error_scans', 0)}\n"
    text += f"• Среднее время сканирования: {stats.get('avg_scan_time', 0)} мс\n\n"
    
    text += "🎟 *БИЛЕТЫ:*\n"
    text += f"• Всего билетов: {stats.get('total_tickets', 0)}\n"
    text += f"• Отсканировано: {stats.get('scanned_tickets', 0)}\n"
    text += f"• Процент сканирования: {round(stats.get('scanned_tickets', 0) / max(stats.get('total_tickets', 1), 1) * 100, 1)}%\n\n"
    
    text += "💾 *КЭШИРОВАНИЕ:*\n"
    text += f"• Попаданий в кэш: {stats.get('cache_hits', 0)}\n"
    text += f"• Промахов: {stats.get('cache_misses', 0)}\n"
    text += f"• Эффективность: {stats.get('cache_hit_rate', 0)}%\n"
    text += f"• Среднее время генерации: {stats.get('avg_generation_time', 0)} мс\n\n"
    
    text += "📊 *QR MANAGER:*\n"
    text += f"• QR-кодов сгенерировано: {qr_manager_stats.get('qr_generated', 0)}\n"
    text += f"• Ошибок генерации: {qr_manager_stats.get('qr_errors', 0)}\n"
    text += f"• Попаданий в кэш: {qr_manager_stats.get('cache_hits', 0)}\n"
    text += f"• Промахов: {qr_manager_stats.get('cache_misses', 0)}\n"
    text += f"• Hit rate: {qr_manager_stats.get('cache_hit_rate', 0):.1f}%\n"
    
    if stats.get('top_scanners'):
        text += "\n🏆 *ТОП СКАНЕРОВ:*\n"
        for i, scanner in enumerate(stats['top_scanners'][:5], 1):
            success_rate = round((scanner['success'] / max(scanner['scans'], 1)) * 100, 1)
            text += f"{i}. @{scanner['username']}: {scanner['scans']} сканирований ({success_rate}% успешных)\n"
    
    if stats.get('recent_scans'):
        text += "\n📋 *ПОСЛЕДНИЕ СКАНИРОВАНИЯ:*\n"
        for scan in stats['recent_scans'][:5]:
            created_at = scan['created_at']
            if isinstance(created_at, str):
                time_str = created_at[11:16]
            else:
                time_str = created_at.strftime('%H:%M')
            
            emoji = "✅" if scan['result'] == 'success' else "⚠️" if scan['result'] == 'warning' else "❌"
            valid_icons = ""
            if scan.get('signature_valid') is not None:
                valid_icons += "🔐" if scan['signature_valid'] else "❌🔐"
            if scan.get('timestamp_valid') is not None:
                valid_icons += "⏱️" if scan['timestamp_valid'] else "❌⏱️"
            
            text += f"{emoji} {time_str} - @{scan['scanner']} - {scan['code']} {valid_icons}\n"
    
    if stats.get('hourly_stats'):
        text += "\n📅 *АКТИВНОСТЬ ПО ЧАСАМ:*\n"
        for hour_stat in stats['hourly_stats'][-8:]:
            text += f"• {hour_stat['hour']}:00 - {hour_stat['scans']} сканирований\n"
    
    keyboard = [
        [InlineKeyboardButton("🔄 Обновить", callback_data="qr_stats_refresh")],
        [InlineKeyboardButton("🧹 Очистить кэш", callback_data="qr_clear_cache")],
        [InlineKeyboardButton("🔙 Назад", callback_data="admin_dashboard")]
    ]
    
    await update.message.reply_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def qr_stats_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "qr_stats_refresh":
        await qr_stats_command(update, context)
    
    elif data == "qr_clear_cache":
        cleared = qr_manager.clear_cache(older_than=3600)
        await query.edit_message_text(
            f"🧹 *Кэш QR-кодов очищен*\n\n"
            f"Удалено файлов: {cleared}",
            parse_mode=ParseMode.MARKDOWN
        )
        await asyncio.sleep(2)
        await qr_stats_command(update, context)

def get_role_selection_keyboard(user_id: int):
    keyboard = []
    
    is_admin = user_id in ADMIN_IDS
    is_promoter = user_id in PROMOTER_IDS
    
    if is_admin:
        keyboard.append([InlineKeyboardButton("⚡️ Войти в админ-панель", callback_data="select_admin")])
    
    if is_promoter:
        keyboard.append([InlineKeyboardButton("👨‍💼 Войти как промоутер", callback_data="select_promoter")])
    
    keyboard.append([InlineKeyboardButton("👤 Пользователь", callback_data="select_user")])
    
    return InlineKeyboardMarkup(keyboard)

def get_main_menu_keyboard(user_role: str = "user"):
    if user_role == "admin":
        keyboard = [
            [InlineKeyboardButton("💰 Узнать цену", callback_data="price_info"),
             InlineKeyboardButton("🎟 Купить билет", callback_data="buy_start")],
            [InlineKeyboardButton("🎪 Событие", callback_data="event_info"),
             InlineKeyboardButton("📋 Мои заказы", callback_data="my_orders")],
            [InlineKeyboardButton("⚡️ Админ-панель", callback_data="admin_menu"),
             InlineKeyboardButton("📊 Панель управления", callback_data="admin_dashboard")],
            [InlineKeyboardButton("📱 Сканировать QR", callback_data="scan_qr_menu")]
        ]
    elif user_role == "promoter":
        keyboard = [
            [InlineKeyboardButton("💰 Узнать цену", callback_data="price_info"),
             InlineKeyboardButton("🎟 Купить билет", callback_data="buy_start")],
            [InlineKeyboardButton("🎪 Событие", callback_data="event_info"),
             InlineKeyboardButton("📋 Мои заказы", callback_data="my_orders")],
            [InlineKeyboardButton("👨‍💼 Панель промоутера", callback_data="promoter_menu"),
             InlineKeyboardButton("📊 Панель управления", callback_data="admin_dashboard")],
            [InlineKeyboardButton("📱 Сканировать QR", callback_data="scan_qr_menu")]
        ]
    else:
        keyboard = [
            [InlineKeyboardButton("💰 Узнать цену", callback_data="price_info"),
             InlineKeyboardButton("🎟 Купить билет", callback_data="buy_start")],
            [InlineKeyboardButton("🎪 Событие", callback_data="event_info"),
             InlineKeyboardButton("📋 Мои заказы", callback_data="my_orders")]
        ]
    
    return InlineKeyboardMarkup(keyboard)

def get_admin_dashboard_keyboard():
    keyboard = [
        [InlineKeyboardButton("📤 Экспорт данных", callback_data="admin_export"),
         InlineKeyboardButton("💾 Создать бэкап", callback_data="admin_backup")],
        [InlineKeyboardButton("📢 Создать рассылку", callback_data="admin_broadcast"),
         InlineKeyboardButton("🎫 Управление промокодами", callback_data="admin_promo_codes")],
        [InlineKeyboardButton("🔄 Обновить", callback_data="admin_dashboard_refresh"),
         InlineKeyboardButton("🔙 В админ-панель", callback_data="admin_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_ticket_type_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎟 Обычный билет", callback_data="ticket_standard")],
        [InlineKeyboardButton("🎩 VIP билет", callback_data="ticket_vip")],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_group_size_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("1", callback_data="size_1"),
            InlineKeyboardButton("2", callback_data="size_2"),
            InlineKeyboardButton("3", callback_data="size_3"),
            InlineKeyboardButton("4", callback_data="size_4")
        ],
        [
            InlineKeyboardButton("5", callback_data="size_5"),
            InlineKeyboardButton("6", callback_data="size_6"),
            InlineKeyboardButton("7", callback_data="size_7"),
            InlineKeyboardButton("8", callback_data="size_8")
        ],
        [
            InlineKeyboardButton("9", callback_data="size_9"),
            InlineKeyboardButton("10", callback_data="size_10"),
            InlineKeyboardButton("10+", callback_data="size_10_plus")
        ],
        [
            InlineKeyboardButton("✏️ Другое число", callback_data="size_custom"),
            InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_admin_keyboard():
    keyboard = [
        [InlineKeyboardButton("📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton("📈 Панель управления", callback_data="admin_dashboard")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="admin_settings")],
        [InlineKeyboardButton("🎪 Редактировать 'Событие'", callback_data="edit_event_info_text")],
        [InlineKeyboardButton("🎫 Управление промокодами", callback_data="admin_promo_codes")],
        [InlineKeyboardButton("🔄 Сбросить статистику", callback_data="admin_reset_stats")],
        [InlineKeyboardButton("📊 Статистика сканирований", callback_data="scan_stats")],
        [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_promoter_keyboard():
    keyboard = [
        [InlineKeyboardButton("📋 Активные заявки", callback_data="promoter_active")],
        [InlineKeyboardButton("⏳ Отложенные", callback_data="promoter_deferred")],
        [InlineKeyboardButton("📈 Панель управления", callback_data="admin_dashboard")],
        [InlineKeyboardButton("📱 Сканировать QR", callback_data="scan_qr_menu")],
        [InlineKeyboardButton("📊 Статистика сканирований", callback_data="scan_stats")],
        [InlineKeyboardButton("🔙 В главное меню", callback_data="back_to_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_scan_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("📱 Начать сканирование", callback_data="scan_qr_start")],
        [InlineKeyboardButton("📊 Статистика сканирований", callback_data="scan_stats")],
        [InlineKeyboardButton("📊 Расширенная статистика QR", callback_data="qr_stats_refresh")],
        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_admin_settings_keyboard():
    keyboard = [
        [InlineKeyboardButton("💰 Изменить цены", callback_data="edit_prices")],
        [InlineKeyboardButton("📞 Изменить контакты", callback_data="edit_contacts")],
        [InlineKeyboardButton("🔄 Сбросить настройки", callback_data="reset_settings")],
        [InlineKeyboardButton("🔙 Назад", callback_data="admin_back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_reset_stats_keyboard():
    keyboard = [
        [InlineKeyboardButton("✅ Да, сбросить всё", callback_data="confirm_reset_all")],
        [InlineKeyboardButton("👥 Сбросить только список гостей", callback_data="confirm_reset_guests")],
        [InlineKeyboardButton("❌ Нет, отмена", callback_data="admin_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_price_edit_keyboard():
    settings = event_settings.get_all_settings()
    keyboard = [
        [InlineKeyboardButton(f"Стандартная: {settings['price_standard']}₽", callback_data="edit_price_standard")],
        [InlineKeyboardButton(f"Групповая: {settings['price_group']}₽", callback_data="edit_price_group")],
        [InlineKeyboardButton(f"VIP: {settings['price_vip']}₽", callback_data="edit_price_vip")],
        [InlineKeyboardButton(f"Порог: {settings['group_threshold']}+ человек", callback_data="edit_group_threshold")],
        [InlineKeyboardButton("🔙 Назад", callback_data="admin_settings")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_contacts_edit_keyboard():
    settings = event_settings.get_all_settings()
    keyboard = [
        [InlineKeyboardButton(f"Telegram: {settings['contact_telegram']}", callback_data="edit_contact_telegram")],
        [InlineKeyboardButton("🔙 Назад", callback_data="admin_settings")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_confirmation_keyboard():
    keyboard = [
        [InlineKeyboardButton("✅ Купить билет", callback_data="confirm_buy")],
        [InlineKeyboardButton("❌ Отменить", callback_data="cancel_buy")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_order_actions_keyboard(order_id: str, user_id: int, username: str = None, is_own_order: bool = False):
    keyboard = []
    
    if not is_own_order:
        if username and username != 'без username' and username != 'None':
            clean_username = username.lstrip('@')
            chat_link = f"https://t.me/{clean_username}"
            keyboard.append([InlineKeyboardButton("💬 Перейти в диалог", url=chat_link)])
        else:
            keyboard.append([InlineKeyboardButton("💬 Перейти в диалог", url=f"tg://user?id={user_id}")])
        
        keyboard.append([InlineKeyboardButton("✅ Закрыть заявку", callback_data=f"close_order_{order_id}")])
        keyboard.append([InlineKeyboardButton("⏳ Отложить", callback_data=f"defer_order_{order_id}")])
        keyboard.append([InlineKeyboardButton("❌ Возврат", callback_data=f"refund_order_{order_id}")])
    else:
        keyboard.append([InlineKeyboardButton("❌ Это ваш заказ, вы не можете его обработать", callback_data="promoter_active")])
    
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="promoter_active")])
    
    return InlineKeyboardMarkup(keyboard)

def get_back_to_promoter_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔙 В меню промоутера", callback_data="promoter_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_promo_management_keyboard():
    keyboard = [
        [InlineKeyboardButton("➕ Создать промокод", callback_data="admin_create_promo")],
        [InlineKeyboardButton("📋 Список промокодов", callback_data="admin_view_promo_list")],
        [InlineKeyboardButton("🔙 Назад", callback_data="admin_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_user_order_actions_keyboard(order_id: str):
    keyboard = [
        [InlineKeyboardButton("🎫 Получить QR-код билета", callback_data=f"get_qr_{order_id}")],
        [InlineKeyboardButton("🔙 Назад", callback_data="my_orders")]
    ]
    return InlineKeyboardMarkup(keyboard)

def format_price_info() -> str:
    settings = event_settings.get_all_settings()
    
    text = (
        f"💰 *Цены на билеты {settings['event_name']}:*\n\n"
        f"• 🎟 *Обычный билет:*\n"
        f"  - 1 человек: *{settings['price_standard']} ₽*\n"
        f"  - Группа от {settings['group_threshold']}+ человек: *{settings['price_group']} ₽/чел.*\n\n"
        f"• 🎩 *VIP билет:*\n"
        f"  - Цена за человека: *{settings['price_vip']} ₽*\n\n"
        f"🎉 *Акция:* Экономия *{settings['price_standard'] - settings['price_group']} ₽* с каждого в группе!\n\n"
        f"Хотите купить билеты?"
    )
    
    return text

def format_price_calculation(group_size: int, ticket_type: str = "standard") -> str:
    settings = event_settings.get_all_settings()
    
    if ticket_type == "vip":
        price_per_person = settings['price_vip']
        total = price_per_person * group_size
        
        text = f"🎩 *Расчет для {group_size} VIP билетов:*\n\n"
        text += f"• Цена за VIP билет: *{price_per_person} ₽*\n"
        text += f"• Общая сумма: *{total} ₽*\n"
        text += f"\n_Цена VIP билета всегда фиксированная: {settings['price_vip']} ₽_"
        
    else:
        if group_size >= settings['group_threshold']:
            price_per_person = settings['price_group']
        else:
            price_per_person = settings['price_standard']
        
        total = price_per_person * group_size
        
        text = f"🎟 *Расчет для {group_size} обычных билетов:*\n\n"
        text += f"• Цена за билет: *{price_per_person} ₽*\n"
        text += f"• Общая сумма: *{total} ₽*\n"
        
        if group_size >= settings['group_threshold']:
            economy = (settings['price_standard'] - settings['price_group']) * group_size
            text += f"\n✅ *Вы получаете групповую скидку!*\n"
            text += f"Экономия: *{economy} ₽*\n"
        
        text += f"\n_Цена для 1 человека: {settings['price_standard']} ₽_\n"
        text += f"_Группа от {settings['group_threshold']}+ человек: {settings['price_group']} ₽/чел._"
    
    return text

def format_order_summary(name: str, email: str, group_size: int, guests: List[str], ticket_type: str = "standard") -> str:
    settings = event_settings.get_all_settings()
    total = event_settings.calculate_price(group_size, ticket_type)
    
    if ticket_type == "vip":
        price_per_person = settings['price_vip']
        ticket_type_text = "VIP 🎩"
    else:
        price_per_person = settings['price_group'] if group_size >= settings['group_threshold'] else settings['price_standard']
        ticket_type_text = "Обычный 🎟"
    
    escaped_name = escape_markdown(str(name))
    escaped_email = escape_markdown(str(email))
    escaped_guests = [escape_markdown(str(guest)) for guest in guests]
    
    summary = "📋 *Сводка вашего заказа:*\n\n"
    summary += f"• Тип билета: *{ticket_type_text}*\n"
    summary += f"• Количество человек: *{group_size}*\n"
    summary += f"• Цена за билет: *{price_per_person} ₽*\n"
    summary += f"• Общая сумма: *{total} ₽*\n\n"
    
    summary += f"• Контактное лицо: *{escaped_name}*\n"
    summary += f"• Email: *{escaped_email}*\n"
    
    if guests:
        summary += "\n• *Список гостей:*\n"
        for i, guest in enumerate(escaped_guests, 1):
            summary += f"  {i}. {guest}\n"
    
    summary += f"\n*Подтвердить покупку?*"
    
    return summary

def format_event_info() -> str:
    event_info_text = event_settings.get_all_settings().get('event_info_text', '')
    
    if event_info_text:
        try:
            return event_info_text
        except Exception as e:
            logger.error(f"Ошибка форматирования event_info_text: {e}")
            return event_info_text
    else:
        settings = event_settings.get_all_settings()
        
        event_name = str(settings.get('event_name', 'SMILE PARTY 🎉'))
        event_address = str(settings.get('event_address', 'Адрес не указан'))
        event_date = str(settings.get('event_date', 'Дата не указана'))
        event_time = str(settings.get('event_time', 'Время не указано'))
        event_age_limit = str(settings.get('event_age_limit', '18+'))
        contact_telegram = str(settings.get('contact_telegram', '@smile_party'))
        
        description = settings.get('description', '')
        if description is None:
            description = ""
        description = str(description)
        
        escaped_name = escape_markdown(event_name)
        escaped_address = escape_markdown(event_address)
        escaped_description = escape_markdown(description)
        
        text = (
            f"🏢 *Информация о мероприятии*\n\n"
            f"*🎉 Название:* {escaped_name}\n"
            f"*📍 Адрес:* {escaped_address}\n"
            f"*📅 Дата:* {event_date}\n"
            f"*⏰ Время:* {event_time}\n"
            f"*🎭 Возраст:* {event_age_limit}\n"
            f"*📱 Telegram:* {contact_telegram}\n"
        )
        
        if escaped_description.strip():
            text += f"\n*📝 Описание:*\n{escaped_description}"
        
        return text

def format_order_details_for_promoter(order: Dict, is_own_order: bool = False) -> str:
    try:
        guests = db.get_order_guests(order['order_id'])
        
        user_name = escape_markdown(str(order['user_name']))
        username = order['username'] if order['username'] else 'без username'
        escaped_username = escape_markdown(username)
        user_email = escape_markdown(str(order['user_email']))
        
        created_at = order['created_at']
        if isinstance(created_at, str):
            created_date = created_at[:16].replace('T', ' ')
        else:
            created_date = created_at.strftime('%d.%m.%Y %H:%M')
        
        formatted_code = format_code_for_display(order['order_code'])
        
        ticket_type_text = "VIP 🎩" if order.get('ticket_type') == 'vip' else "Обычный 🎟"
        
        text = (
            f"📋 *Детали заказа #{order['order_id']}*\n\n"
            f"*🔑 Уникальный код:* `{order['order_code']}`\n"
            f"*🎫 Тип билета:* {ticket_type_text}\n\n"
            f"👤 *Контактное лицо:* {user_name}\n"
            f"📱 *Telegram:* @{escaped_username}\n"
            f"📧 *Email:* {user_email}\n"
            f"🆔 *User ID:* `{order['user_id']}`\n"
            f"👥 *Количество человек:* {order['group_size']}\n"
            f"💰 *Сумма заказа:* {order['total_amount']} ₽\n"
            f"📅 *Дата создания:* {created_date}\n"
            f"📊 *Статус:* {order['status']}"
        )
        
        if order.get('assigned_promoter'):
            assigned_promoter = escape_markdown(str(order['assigned_promoter']))
            text += f"\n👨‍💼 *Назначен:* @{assigned_promoter}"
        
        if order.get('scanned_at'):
            scanned_at = order['scanned_at']
            if isinstance(scanned_at, str):
                scanned_time = scanned_at[:16]
            else:
                scanned_time = scanned_at.strftime('%d.%m.%Y %H:%M')
            text += f"\n📱 *Отсканирован:* {scanned_time} (@{order.get('scanned_by', 'неизвестно')})"
        
        if guests:
            text += f"\n\n📝 *Список гостей:*"
            for guest in guests:
                guest_name = escape_markdown(str(guest['full_name']))
                guest_scanned = "✅" if guest.get('scanned_at') else "⏳"
                text += f"\n{guest_scanned} {guest_name}"
        
        text += f"\n\n*💬 Способы связи:*"
        
        if username and username != 'без username' and username != 'None':
            clean_username = username.lstrip('@')
            text += f"\n• Telegram: @{clean_username}"
            text += f"\n• Ссылка: https://t.me/{clean_username}"
        else:
            text += f"\n• User ID: {order['user_id']}"
            text += f"\n• Ссылка: tg://user?id={order['user_id']}"
        
        text += f"\n• Email: {user_email}"
        
        if is_own_order:
            text += f"\n\n⚠️ *ВНИМАНИЕ:* Это ваш собственный заказ! Вы не можете его обработать."
        
        return text
    except Exception as e:
        logger.error(f"Ошибка при форматировании деталей заказа: {e}")
        return f"📋 *Детали заказа #{order['order_id']}*\n\n👤 *Контакт:* {escape_markdown(str(order['user_name']))}\n💰 *Сумма:* {order['total_amount']} ₽"

def format_statistics() -> str:
    stats = db.get_statistics()
    scan_stats = db.get_scan_stats()
    
    text = (
        "📊 *Статистика*\n\n"
        f"📋 *Всего заказов:* {stats.get('total_orders', 0)}\n"
        f"🟢 *Активные:* {stats.get('active_orders', 0)}\n"
        f"⏳ *Отложенные:* {stats.get('deferred_orders', 0)}\n"
        f"✅ *Закрытые:* {stats.get('closed_orders', 0)}\n"
        f"❌ *Возвраты:* {stats.get('refunded_orders', 0)}\n"
        f"💰 *Выручка:* {stats.get('revenue', 0)} ₽\n"
        f"👥 *Всего гостей в списках:* {stats.get('total_guests', 0)}\n\n"
        f"🎟 *Обычные билеты:*\n"
        f"• Продано: {stats.get('standard_tickets', 0)}\n"
        f"• Выручка: {stats.get('standard_revenue', 0)} ₽\n\n"
        f"🎩 *VIP билеты:*\n"
        f"• Продано: {stats.get('vip_tickets', 0)}\n"
        f"• Выручка: {stats.get('vip_revenue', 0)} ₽\n\n"
        f"📅 *Сегодня:*\n"
        f"• Заказов: {stats.get('today_orders', 0)}\n"
        f"• Выручка: {stats.get('today_revenue', 0)} ₽\n"
        f"• Покупателей: {stats.get('today_users', 0)}\n\n"
        f"📱 *Статистика сканирований:*\n"
        f"• Всего сканирований: {scan_stats.get('total_scans', 0)}\n"
        f"• Успешных: {scan_stats.get('success_scans', 0)}\n"
        f"• Повторных: {scan_stats.get('warning_scans', 0)}\n"
        f"• Отсканировано билетов: {scan_stats.get('scanned_tickets', 0)}/{scan_stats.get('total_valid_tickets', 0)}\n"
        f"• Сегодня: {scan_stats.get('today_scans', 0)} (успешно: {scan_stats.get('today_success', 0)})"
    )
    
    return text

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        user = update.effective_user
        message_text = update.message.text
        
        if not rate_limiter.check_limit(user.id):
            remaining = rate_limiter.get_remaining(user.id)
            await update.message.reply_text(
                f"⏰ *Слишком много запросов!*\n\n"
                f"Пожалуйста, подождите. Доступно запросов через 5 секунд. {remaining}",
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
        
        db.add_user(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        db.update_user_request(user.id)
        log_user_action(user.id, "start_command")
        
        context.user_data.clear()
        
        if ' ' in message_text:
            params = message_text.split(' ', 1)[1]
            
            if params.startswith('order_'):
                order_id = params.replace('order_', '')
                order = db.get_order(order_id)
                
                if order and user.id in PROMOTER_IDS:
                    own_order = is_own_order(order, user.id)
                    
                    if own_order:
                        await update.message.reply_text(
                            "❌ *Это ваш собственный заказ!*\n\n"
                            "Вы не можете обрабатывать свой собственный заказ.\n"
                            "Пожалуйста, выберите другой заказ для обработки.",
                            parse_mode=ParseMode.MARKDOWN
                        )
                    else:
                        username = user.username or f"user_{user.id}"
                        context.user_data['user_role'] = 'promoter'
                        
                        text = format_order_details_for_promoter(order, own_order)
                        username_for_link = order['username'] if order['username'] and order['username'] != 'без username' and order['username'] != 'None' else None
                        
                        await update.message.reply_text(
                            text,
                            reply_markup=get_order_actions_keyboard(order_id, order['user_id'], username_for_link, own_order),
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return PROMOTER_VIEW_ORDER
        
        role = get_user_role(user.id)
        
        if role == "admin" or role == "promoter":
            settings_data = event_settings.get_all_settings()
            await update.message.reply_text(
                f"🎉 *Добро пожаловать в {escape_markdown(str(settings_data['event_name']))}!*\n\n"
                f"Пожалуйста, выберите, как вы хотите войти:",
                reply_markup=get_role_selection_keyboard(user.id),
                parse_mode=ParseMode.MARKDOWN
            )
            return ROLE_SELECTION
        else:
            context.user_data['user_role'] = 'user'
            settings_data = event_settings.get_all_settings()
            await update.message.reply_text(
                f"🎉 *Добро пожаловать в {escape_markdown(str(settings_data['event_name']))}!*\n\n"
                f"Выберите действие:",
                reply_markup=get_main_menu_keyboard('user'),
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
    except Exception as e:
        logger.error(f"Ошибка в start_command: {e}")
        await update.message.reply_text("❌ Произошла ошибка при запуске бота.")
        return MAIN_MENU

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    username = query.from_user.username or f"user_{user_id}"
    data = query.data
    
    if not rate_limiter.check_limit(user_id):
        remaining = rate_limiter.get_remaining(user_id)
        await query.edit_message_text(
            f"⏰ *Слишком много запросов!*\n\n"
            f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    db.update_user_request(user_id)
    
    try:
        if data.startswith("select_"):
            role = data.replace("select_", "")
            
            if role == "admin" and user_id not in ADMIN_IDS:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_role_selection_keyboard(user_id),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ROLE_SELECTION
            
            if role == "promoter" and user_id not in PROMOTER_IDS:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_role_selection_keyboard(user_id),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ROLE_SELECTION
            
            context.user_data['user_role'] = role
            
            if role == "admin":
                await query.edit_message_text(
                    "⚡️ *Вы вошли как администратор*\n\n"
                    "Выберите действие:",
                    reply_markup=get_main_menu_keyboard(role),
                    parse_mode=ParseMode.MARKDOWN
                )
            elif role == "promoter":
                await query.edit_message_text(
                    "👨‍💼 *Вы вошли как промоутер*\n\n"
                    "Выберите действие:",
                    reply_markup=get_main_menu_keyboard(role),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await query.edit_message_text(
                    "👤 *Вы вошли как пользователь*\n\n"
                    "Выберите действие:",
                    reply_markup=get_main_menu_keyboard(role),
                    parse_mode=ParseMode.MARKDOWN
                )
            
            return MAIN_MENU
        
        elif data == "price_info":
            await query.edit_message_text(
                format_price_info(),
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
        
        elif data == "event_info":
            try:
                text = format_event_info()
                
                try:
                    await query.edit_message_text(
                        text,
                        reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                        parse_mode=ParseMode.MARKDOWN
                    )
                except BadRequest as e:
                    logger.error(f"Ошибка при отправке Markdown: {e}")
                    plain_text = text.replace('*', '').replace('_', '').replace('`', '')
                    await query.edit_message_text(
                        plain_text,
                        reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user'))
                    )
                
            except Exception as e:
                logger.error(f"Ошибка при отображении информации о мероприятии: {e}")
                settings_data = event_settings.get_all_settings()
                simple_text = (
                    f"🏢 Информация о мероприятии\n\n"
                    f"🎉 Название: {settings_data.get('event_name', 'SMILE PARTY')}\n"
                    f"📍 Адрес: {settings_data.get('event_address', 'Адрес не указан')}\n"
                    f"📅 Дата: {settings_data.get('event_date', 'Дата не указана')}\n"
                    f"⏰ Время: {settings_data.get('event_time', 'Время не указано')}\n"
                    f"🎭 Возраст: {settings_data.get('event_age_limit', '18+')}\n"
                    f"📱 Telegram: {settings_data.get('contact_telegram', '@smile_party')}"
                )
                
                await query.edit_message_text(
                    simple_text,
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user'))
                )
            
            return MAIN_MENU
        
        elif data == "my_orders":
            orders = db.get_user_orders(user_id)
            
            if not orders:
                keyboard = [
                    [InlineKeyboardButton("🎟 Купить билет", callback_data="buy_start")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")]
                ]
                
                await query.edit_message_text(
                    "📭 *У вас пока нет заказов*\n\n"
                    "Хотите купить билет?",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                orders_text = "📋 *Ваши заказы:*\n\n"
                for order in orders[:10]:
                    status_emoji = {
                        "active": "🟢",
                        "deferred": "⏳",
                        "closed": "✅",
                        "refunded": "❌"
                    }.get(order["status"], "❓")
                    
                    ticket_type_emoji = "🎩" if order.get('ticket_type') == 'vip' else "🎟"
                    
                    created_at = order['created_at']
                    if isinstance(created_at, str):
                        created_date = created_at[:10]
                    else:
                        created_date = created_at.strftime('%d.%m.%Y')
                    
                    formatted_code = format_code_for_display(order.get('order_code', 'НЕТ КОДА'))
                    
                    orders_text += (
                        f"{status_emoji} *Заказ #{order['order_id']}* {ticket_type_emoji}\n"
                        f"🔑 Код: `{order.get('order_code', 'НЕТ КОДА')}`\n"
                        f"👥 {order['group_size']} чел. | "
                        f"💰 {order['total_amount']} ₽ | "
                        f"📅 {created_date}\n"
                        f"Статус: {order['status']}\n\n"
                    )
                
                if len(orders_text) > 4096:
                    orders_text = orders_text[:4000] + "...\n\n⚠️ Слишком много заказов, показаны только последние."
                
                keyboard_buttons = []
                for order in orders[:5]:
                    if order['status'] == 'closed':
                        keyboard_buttons.append([
                            InlineKeyboardButton(
                                f"🎫 QR для #{order['order_id']}", 
                                callback_data=f"get_qr_{order['order_id']}"
                            )
                        ])
                
                keyboard_buttons.append([
                    InlineKeyboardButton("🎟 Новый заказ", callback_data="buy_start"),
                    InlineKeyboardButton("🔙 Назад", callback_data="back_to_menu")
                ])
                
                await query.edit_message_text(
                    orders_text,
                    reply_markup=InlineKeyboardMarkup(keyboard_buttons),
                    parse_mode=ParseMode.MARKDOWN
                )
            
            return MAIN_MENU
        
        elif data.startswith("get_qr_"):
            order_id = data.replace("get_qr_", "")
            order = db.get_order(order_id)
            
            if order and order['user_id'] == user_id:
                await generate_ticket_qr(update, context, order['order_code'])
                return MAIN_MENU
            else:
                await query.edit_message_text(
                    "❌ *Заказ не найден или у вас нет прав для просмотра*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "back_to_menu":
            role = context.user_data.get('user_role', 'user')
            await query.edit_message_text(
                f"🏠 *Главное меню*\n\n"
                f"Выберите действие:",
                reply_markup=get_main_menu_keyboard(role),
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
        
        elif data == "buy_start":
            await query.edit_message_text(
                "🎫 *Покупка билета*\n\n"
                "Сначала выберите тип билета:",
                reply_markup=get_ticket_type_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
            return BUY_TICKET_TYPE
        
        elif data in ["ticket_standard", "ticket_vip"]:
            if data == "ticket_standard":
                context.user_data['ticket_type'] = 'standard'
                ticket_type_text = "обычный"
            else:
                context.user_data['ticket_type'] = 'vip'
                ticket_type_text = "VIP"
            
            await query.edit_message_text(
                f"🎟 *Покупка {ticket_type_text} билета*\n\n"
                "Теперь выберите количество человек:",
                reply_markup=get_group_size_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
            return BUY_TICKET_TYPE
        
        elif data.startswith("size_"):
            size_data = data.replace("size_", "")
            
            if size_data == "custom":
                await query.edit_message_text(
                    "✏️ *Введите количество человек цифрами*\n\n"
                    "Можно указать любое число от 1 до 100\n"
                    "Например: 15, 25, 50",
                    parse_mode=ParseMode.MARKDOWN
                )
                return BUY_TICKET_TYPE
            
            elif size_data == "10_plus":
                context.user_data['group_size'] = 15
                await query.edit_message_text(
                    "✏️ *Введите количество человек цифрами*\n\n"
                    "Можно указать любое число от 10 до 100\n"
                    "Например: 12, 20, 45",
                    parse_mode=ParseMode.MARKDOWN
                )
                return BUY_TICKET_TYPE
            else:
                try:
                    group_size = int(size_data)
                except:
                    group_size = 1
            
            context.user_data['group_size'] = group_size
            context.user_data['guests'] = []
            
            ticket_type = context.user_data.get('ticket_type', 'standard')
            
            await query.edit_message_text(
                format_price_calculation(group_size, ticket_type) + "\n\n"
                "👉 *Продолжить покупку?*",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Да, продолжить", callback_data="buy_continue")],
                    [InlineKeyboardButton("❌ Нет, отмена", callback_data="back_to_menu")]
                ]),
                parse_mode=ParseMode.MARKDOWN
            )
            return BUY_TICKET_TYPE
        
        elif data == "buy_continue":
            context.user_data['in_buy_process'] = True
            
            await query.edit_message_text(
                "👤 *Введите ваше имя и фамилию (контактное лицо)*\n\n"
                "Например: Александр Иванов",
                parse_mode=ParseMode.MARKDOWN
            )
            return BUY_NAME
        
        elif data == "confirm_buy":
            required_fields = ['name', 'email', 'group_size', 'guests', 'ticket_type']
            if not all(field in context.user_data for field in required_fields):
                await query.edit_message_text(
                    "❌ *Ошибка: недостаточно данных*\n\n"
                    "Пожалуйста, начните покупку заново.",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
            
            current_hour = datetime.now().hour
            is_night_time = current_hour >= 23 or current_hour < 8
            
            total_amount = event_settings.calculate_price(
                context.user_data['group_size'], 
                context.user_data['ticket_type']
            )
            
            order_data = db.create_order(
                user_id=user_id,
                username=username,
                user_name=context.user_data['name'],
                user_email=context.user_data['email'],
                group_size=context.user_data['group_size'],
                ticket_type=context.user_data['ticket_type'],
                total_amount=total_amount
            )
            
            if not order_data:
                await query.edit_message_text(
                    "❌ *Ошибка при создании заказа*\n\n"
                    "Пожалуйста, попробуйте еще раз.",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
            
            order_id = order_data['order_id']
            order_code = order_data['order_code']
            formatted_code = format_code_for_display(order_code)
            
            if not db.add_guests_to_order(order_id, order_code, context.user_data['guests']):
                await query.edit_message_text(
                    "❌ *Ошибка при добавлении гостей*\n\n"
                    "Заказ создан, но возникла проблема с сохранением списка гостей.",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
            
            settings_data = event_settings.get_all_settings()
            
            ticket_type_text = "VIP" if context.user_data['ticket_type'] == 'vip' else "Обычный"
            
            confirmation_text = (
                f"🎉 ЗАКАЗ #{order_id} УСПЕШНО СОЗДАН!\n\n"
                f"*🎫 Тип билета:* {ticket_type_text}\n"
                f"*🔑 Ваш уникальный код:* `{order_code}`\n\n"
                f"👤 Контактное лицо: {escape_markdown(str(context.user_data['name']))}\n"
                f"📧 Email: {escape_markdown(str(context.user_data['email']))}\n"
                f"👥 Количество: {context.user_data['group_size']} чел.\n"
                f"💰 Сумма: {total_amount} ₽\n\n"
                f"*💾 Сохраните ваш код! Он потребуется при входе на мероприятие.*\n\n"
            )
            
            if is_night_time:
                confirmation_text += (
                    "⏰ ВНИМАНИЕ! Вы оформили заказ в нерабочее время (23:00 - 08:00).\n"
                    "Промоутеры свяжутся с вами утром для подтверждения.\n\n"
                )
            else:
                confirmation_text += (
                    "ЧТО ДАЛЬШЕ?\n"
                    "1. Все гости добавлены в списки на вход\n"
                    "2. В течение 30 минут с вами свяжется промоутер\n"
                    "3. Он подтвердит покупку\n\n"
                )
            
            confirmation_text += f"СПАСИБО ЗА ПОКУПКУ В {settings_data['event_name']}! 🎊"
            
            await query.message.reply_text(confirmation_text, parse_mode=ParseMode.MARKDOWN)
            
            order = db.get_order(order_id)
            if order:
                await send_new_order_notification(context, order)
            
            context.user_data.pop('in_buy_process', None)
            context.user_data.pop('name', None)
            context.user_data.pop('email', None)
            context.user_data.pop('group_size', None)
            context.user_data.pop('guests', None)
            context.user_data.pop('guest_counter', None)
            context.user_data.pop('ticket_type', None)
            
            await query.message.reply_text(
                "Выберите дальнейшее действие:",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user'))
            )
            
            return MAIN_MENU
        
        elif data == "cancel_buy":
            context.user_data.pop('in_buy_process', None)
            context.user_data.pop('name', None)
            context.user_data.pop('email', None)
            context.user_data.pop('group_size', None)
            context.user_data.pop('guests', None)
            context.user_data.pop('guest_counter', None)
            context.user_data.pop('ticket_type', None)
            
            await query.edit_message_text(
                "❌ *Покупка отменена*\n\n"
                "Если передумаете — всегда можете создать новый заказ!",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
            
            return MAIN_MENU
        
        elif data == "admin_menu":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "⚡️ *Панель администратора*\n\n"
                    "Выберите действие:",
                    reply_markup=get_admin_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_back":
            await query.edit_message_text(
                "⚡️ *Панель администратора*",
                reply_markup=get_admin_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
            return ADMIN_MENU
        
        elif data == "admin_stats":
            if user_id in ADMIN_IDS:
                stats_text = format_statistics()
                await query.edit_message_text(
                    stats_text,
                    reply_markup=get_admin_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_dashboard":
            if user_id in ADMIN_IDS or user_id in PROMOTER_IDS:
                return await dashboard_command(update, context)
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав для просмотра панели управления*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_dashboard_refresh":
            return await dashboard_command(update, context)
        
        elif data == "admin_export":
            if user_id in ADMIN_IDS:
                await export_command(update, context)
                return ADMIN_DASHBOARD
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_backup":
            if user_id in ADMIN_IDS:
                await backup_command(update, context)
                return ADMIN_DASHBOARD
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_broadcast":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "📢 *Создание рассылки*\n\n"
                    "Введите сообщение для рассылки пользователям:\n\n"
                    "Используйте команду /broadcast <текст сообщения>",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_BROADCAST
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_promo_codes":
            if user_id in ADMIN_IDS:
                return await promo_manage_command(update, context)
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_create_promo":
            if user_id in ADMIN_IDS:
                context.user_data['creating_promo'] = True
                context.user_data['promo_step'] = 'code'
                
                await query.edit_message_text(
                    "🎫 *Создание промокода*\n\n"
                    "Шаг 1/4: Введите код промокода (только латинские буквы и цифры):\n\n"
                    "Пример: SMILE2024, PARTY50, DISCOUNT100",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_CREATE_PROMO
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_view_promo":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "🎫 *Просмотр промокода*\n\n"
                    "Введите код промокода для просмотра деталей:",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_VIEW_PROMO
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "admin_view_promo_list":
            return await promo_manage_command(update, context)
        
        elif data == "admin_reset_stats":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "🔄 *Сброс статистики*\n\n"
                    "⚠️ *ВНИМАНИЕ!* Это действие удалит:\n"
                    "• Все заказы\n"
                    "• Всех гостей\n"
                    "• Всю историю\n\n"
                    "Выберите действие:",
                    reply_markup=get_reset_stats_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_RESET_STATS
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "confirm_reset_all":
            if user_id in ADMIN_IDS:
                with closing(db.get_connection()) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM guests")
                    cursor.execute("DELETE FROM orders")
                    cursor.execute("DELETE FROM promo_codes")
                    cursor.execute("DELETE FROM action_logs")
                    cursor.execute("DELETE FROM scan_logs")
                    cursor.execute("DELETE FROM scan_attempts")
                    cursor.execute("DELETE FROM qr_cache_stats")
                    conn.commit()
                
                await query.edit_message_text(
                    "✅ *Вся статистика успешно сброшена!*\n\n"
                    "Все заказы, гости, промокоды и логи удалены.",
                    reply_markup=get_admin_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await query.edit_message_text(
                    "❌ *Ошибка при сбросе статистики*",
                    reply_markup=get_admin_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
            return ADMIN_MENU
        
        elif data == "confirm_reset_guests":
            if user_id in ADMIN_IDS and db.reset_guests_count():
                await query.edit_message_text(
                    "✅ *Список гостей успешно сброшен!*\n\n"
                    "Все гости удалены из базы данных.",
                    reply_markup=get_admin_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await query.edit_message_text(
                    "❌ *Ошибка при сбросе списка гостей*",
                    reply_markup=get_admin_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
            return ADMIN_MENU
        
        elif data == "admin_settings":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "⚙️ *Настройки мероприятия*\n\n"
                    "Выберите, что хотите изменить:",
                    reply_markup=get_admin_settings_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_EDIT
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "edit_prices":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "💰 *Редактирование цен*\n\n"
                    "Выберите настройку для изменения:",
                    reply_markup=get_price_edit_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_EDIT
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "edit_contacts":
            if user_id in ADMIN_IDS:
                await query.edit_message_text(
                    "📞 *Редактирование контактов*\n\n"
                    "Выберите настройку для изменения:",
                    reply_markup=get_contacts_edit_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_EDIT
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "edit_event_info_text":
            if user_id in ADMIN_IDS:
                context.user_data['editing_key'] = "event_info_text"
                context.user_data['editing_name'] = "текст кнопки 'Событие'"
                
                current_text = event_settings.get_all_settings().get('event_info_text', '')
                if current_text:
                    display_text = current_text
                else:
                    display_text = ""
                
                if len(display_text) > 2000:
                    display_text = display_text[:2000] + "...\n\n[текст слишком длинный, показаны первые 2000 символов]"
                
                await query.edit_message_text(
                    f"✏️ Редактирование текста кнопки 'Событие'\n\n"
                    f"Текущий текст:\n\n{display_text}\n\n"
                    f"Введите новый текст (можно использовать Markdown форматирование, например *жирный* или _курсив_):",
                    parse_mode=None
                )
                return ADMIN_EDIT_TEXT
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "reset_settings":
            if user_id in ADMIN_IDS:
                keyboard = [
                    [InlineKeyboardButton("✅ Да, сбросить", callback_data="confirm_reset_settings")],
                    [InlineKeyboardButton("❌ Нет, отмена", callback_data="admin_settings")]
                ]
                
                await query.edit_message_text(
                    "🔄 *Сброс настроек*\n\n"
                    "Вы уверены, что хотите сбросить все настройки к значениям по умолчанию?\n\n"
                    "⚠️ *Это действие нельзя отменить!*",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_EDIT
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "confirm_reset_settings":
            if user_id in ADMIN_IDS and event_settings.reset_to_defaults():
                await query.edit_message_text(
                    "✅ *Настройки сброшены к значениям по умолчанию!*",
                    reply_markup=get_admin_settings_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await query.edit_message_text(
                    "❌ *Ошибка при сбросе настроек*",
                    reply_markup=get_admin_settings_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
            return ADMIN_EDIT
        
        elif data.startswith("edit_"):
            if user_id in ADMIN_IDS:
                setting_map = {
                    "edit_price_standard": ("стандартную цену (1 человек)", "price_standard"),
                    "edit_price_group": ("групповую цену", "price_group"),
                    "edit_price_vip": ("VIP цену", "price_vip"),
                    "edit_group_threshold": ("порог для групповой цены", "group_threshold"),
                    "edit_contact_telegram": ("контакт в Telegram", "contact_telegram")
                }
                
                if data in setting_map:
                    setting_name, setting_key = setting_map[data]
                    current_value = event_settings.get_all_settings().get(setting_key, "")
                    
                    context.user_data['editing_key'] = setting_key
                    context.user_data['editing_name'] = setting_name
                    
                    await query.edit_message_text(
                        f"✏️ *Редактирование {setting_name}*\n\n"
                        f"Текущее значение: *{current_value}*\n\n"
                        f"Введите новое значение:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_EDIT_TEXT
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав администратора*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "promoter_menu":
            if user_id in PROMOTER_IDS:
                await query.edit_message_text(
                    "👨‍💼 *Панель промоутера*\n\n"
                    "Выберите действие:",
                    reply_markup=get_promoter_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "promoter_active":
            if user_id in PROMOTER_IDS:
                active_orders = db.get_orders_by_status("active")
                
                filtered_orders = []
                for order in active_orders:
                    if not is_own_order(order, user_id):
                        filtered_orders.append(order)
                
                if not filtered_orders:
                    await query.edit_message_text(
                        "✅ *Нет доступных активных заявок*\n\n"
                        "Ваши собственные заказы не отображаются в этом списке.",
                        reply_markup=get_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    keyboard_buttons = []
                    for order in filtered_orders[:10]:
                        formatted_code = format_code_for_display(order.get('order_code', 'НЕТ КОДА'))
                        ticket_type_emoji = "🎩" if order.get('ticket_type') == 'vip' else "🎟"
                        keyboard_buttons.append([
                            InlineKeyboardButton(
                                f"{ticket_type_emoji} {escape_markdown(str(order['user_name']))} - {formatted_code} - {order['total_amount']}₽", 
                                callback_data=f"view_order_{order['order_id']}"
                            )
                        ])
                    
                    keyboard_buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="promoter_menu")])
                    
                    await query.edit_message_text(
                        f"🟢 *Доступные активные заявки:* {len(filtered_orders)}\n\n"
                        "Ваши собственные заказы скрыты из этого списка.\n"
                        "Выберите заявку для обработки:",
                        reply_markup=InlineKeyboardMarkup(keyboard_buttons),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "promoter_deferred":
            if user_id in PROMOTER_IDS:
                deferred_orders = db.get_orders_by_status("deferred")
                
                filtered_orders = []
                for order in deferred_orders:
                    if not is_own_order(order, user_id):
                        filtered_orders.append(order)
                
                if not filtered_orders:
                    await query.edit_message_text(
                        "✅ *Нет доступных отложенных заявки*\n\n"
                        "Ваши собственные заказы не отображаются в этом списках.",
                        reply_markup=get_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    keyboard_buttons = []
                    for order in filtered_orders[:10]:
                        formatted_code = format_code_for_display(order.get('order_code', 'НЕТ КОДА'))
                        ticket_type_emoji = "🎩" if order.get('ticket_type') == 'vip' else "🎟"
                        keyboard_buttons.append([
                            InlineKeyboardButton(
                                f"{ticket_type_emoji} {escape_markdown(str(order['user_name']))} - {formatted_code} - {order['total_amount']}₽", 
                                callback_data=f"activate_order_{order['order_id']}"
                            )
                        ])
                    
                    keyboard_buttons.append([InlineKeyboardButton("🔙 Назад", callback_data="promoter_menu")])
                    
                    await query.edit_message_text(
                        f"⏳ *Доступные отложенные заявки:* {len(filtered_orders)}\n\n"
                        "Ваши собственные заказы скрыты из этого списка.\n"
                        "Выберите заявку для активации:",
                        reply_markup=InlineKeyboardMarkup(keyboard_buttons),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                return PROMOTER_DEFERRED
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data.startswith("view_order_"):
            if user_id in PROMOTER_IDS:
                order_id = data.replace("view_order_", "")
                order = db.get_order(order_id)
                
                if order:
                    own_order = is_own_order(order, user_id)
                    text = format_order_details_for_promoter(order, own_order)
                    
                    try:
                        username_for_link = order['username'] if order['username'] and order['username'] != 'без username' and order['username'] != 'None' else None
                        await query.edit_message_text(
                            text,
                            reply_markup=get_order_actions_keyboard(order_id, order['user_id'], username_for_link, own_order),
                            parse_mode=ParseMode.MARKDOWN
                        )
                    except BadRequest:
                        plain_text = text.replace('*', '').replace('_', '').replace('`', '')
                        await query.edit_message_text(
                            plain_text,
                            reply_markup=get_order_actions_keyboard(order_id, order['user_id'], username_for_link, own_order)
                        )
                    
                    return PROMOTER_VIEW_ORDER
                else:
                    await query.edit_message_text(
                        "❌ *Заказ не найден*",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data.startswith("activate_order_"):
            if user_id in PROMOTER_IDS:
                order_id = data.replace("activate_order_", "")
                order = db.get_order(order_id)
                
                if order and is_own_order(order, user_id):
                    await query.edit_message_text(
                        "❌ *Вы не можете активировать свой собственный заказ!*\n\n"
                        "Пожалуйста, выберите другой заказ для обработки.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return PROMOTER_MENU
                
                if db.update_order_status(order_id, "active", username):
                    await query.edit_message_text(
                        f"✅ *Заказ #{order_id} активирован!*\n\n"
                        f"Заявка перемещена в активные.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_text(
                        "❌ *Ошибка при активации заказа*",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data.startswith("close_order_"):
            if user_id in PROMOTER_IDS:
                order_id = data.replace("close_order_", "")
                order = db.get_order(order_id)
                
                if order and is_own_order(order, user_id):
                    await query.edit_message_text(
                        "❌ *Вы не можете закрыть свой собственный заказ!*\n\n"
                        "Пожалуйста, выберите другой заказ для обработки.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return PROMOTER_MENU
                
                if db.update_order_status(order_id, "closed", username):
                    await send_channel_notification(context, order, username, "closed")
                    
                    await send_to_lists_channel(context, order, username)
                    
                    await send_order_notification_to_user(context, order, "closed", username)
                    
                    db.mark_order_processed(order_id)
                    
                    await query.edit_message_text(
                        f"✅ *Заказ #{order_id} успешно закрыт!*\n\n"
                        f"Уведомления отправлены:\n"
                        f"• В канал закрытых заявок\n"
                        f"• В канал со списками\n"
                        f"• Пользователю (с QR-кодом)",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_text(
                        "❌ *Ошибка при закрытии заказа*",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data.startswith("defer_order_"):
            if user_id in PROMOTER_IDS:
                order_id = data.replace("defer_order_", "")
                order = db.get_order(order_id)
                
                if order and is_own_order(order, user_id):
                    await query.edit_message_text(
                        "❌ *Вы не можете отложить свой собственный заказ!*\n\n"
                        "Пожалуйста, выберите другой заказ для обработки.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return PROMOTER_MENU
                
                if db.update_order_status(order_id, "deferred", username):
                    await query.edit_message_text(
                        f"⏳ *Заказ #{order_id} отложен!*\n\n"
                        f"Заявка перемещена в раздел отложенных.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_text(
                        "❌ *Ошибка при откладывании заказа*",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data.startswith("refund_order_"):
            if user_id in PROMOTER_IDS:
                order_id = data.replace("refund_order_", "")
                order = db.get_order(order_id)
                
                if order and is_own_order(order, user_id):
                    await query.edit_message_text(
                        "❌ *Вы не можете оформить возврат на свой собственный заказ!*\n\n"
                        "Пожалуйста, выберите другой заказ для обработки.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return PROMOTER_MENU
                
                if db.update_order_status(order_id, "refunded", username):
                    await send_channel_notification(context, order, username, "refunded")
                    
                    await send_order_notification_to_user(context, order, "refunded", username)
                    
                    await query.edit_message_text(
                        f"❌ *Возврат по заказу #{order_id} оформлен!*\n\n"
                        f"Уведомления отправлены в канал и пользователю.",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_text(
                        "❌ *Ошибка при оформлении возврата*",
                        reply_markup=get_back_to_promoter_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                return PROMOTER_MENU
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав промоутера*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "change_role":
            await query.edit_message_text(
                "🔄 *Смена роли*\n\n"
                "Пожалуйста, выберите, как вы хотите войти:",
                reply_markup=get_role_selection_keyboard(user_id),
                parse_mode=ParseMode.MARKDOWN
            )
            return ROLE_SELECTION
        
        elif data == "scan_qr_menu":
            if user_id in SCANNER_IDS:
                await query.edit_message_text(
                    "📱 *Меню сканирования QR-кодов*\n\n"
                    "Выберите действие:",
                    reply_markup=get_scan_menu_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return SCAN_QR
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав для сканирования QR-кодов*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "scan_qr_start":
            if user_id in SCANNER_IDS:
                context.user_data['scan_mode'] = True
                await query.edit_message_text(
                    "📱 *Сканирование QR-кода*\n\n"
                    "Пожалуйста, отправьте фото QR-кода билета для проверки.\n\n"
                    "Или введите код билета вручную (например: #KA123456)",
                    parse_mode=ParseMode.MARKDOWN
                )
                return SCAN_QR
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав для сканирования QR-кодов*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "scan_stats":
            if user_id in SCANNER_IDS or user_id in ADMIN_IDS:
                stats = db.get_scan_stats()
                
                text = "📊 *Статистика сканирований QR-кодов*\n\n"
                text += f"• Всего сканирований: {stats.get('total_scans', 0)}\n"
                text += f"• ✅ Успешных: {stats.get('success_scans', 0)}\n"
                text += f"• ⚠️ Повторных: {stats.get('warning_scans', 0)}\n"
                text += f"• ❌ Ошибок: {stats.get('error_scans', 0)}\n"
                text += f"• 📱 Отсканировано билетов: {stats.get('scanned_tickets', 0)}/{stats.get('total_valid_tickets', 0)}\n"
                text += f"• 📅 Сегодня: {stats.get('today_scans', 0)} (успешно: {stats.get('today_success', 0)})\n\n"
                
                if stats.get('top_scanners'):
                    text += "🏆 *Топ сканеров:*\n"
                    for i, scanner in enumerate(stats['top_scanners'][:5], 1):
                        text += f"{i}. @{scanner['scanner_username']}: {scanner['scan_count']} сканирований\n"
                    text += "\n"
                
                if stats.get('recent_scans'):
                    text += "📋 *Последние 5 сканирований:*\n"
                    for scan in stats['recent_scans'][:5]:
                        created_at = scan['created_at']
                        if isinstance(created_at, str):
                            time_str = created_at[11:16]
                        else:
                            time_str = created_at.strftime('%H:%M')
                        
                        emoji = "✅" if scan['scan_result'] == 'success' else "⚠️" if scan['scan_result'] == 'warning' else "❌"
                        text += f"{emoji} {time_str} - @{scan['scanner_username']} - {scan['order_code']}\n"
                
                keyboard = [
                    [InlineKeyboardButton("📱 Сканировать еще", callback_data="scan_qr_start")],
                    [InlineKeyboardButton("🔙 Назад", callback_data="scan_qr_menu")]
                ]
                
                await query.edit_message_text(
                    text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
                return SCAN_QR
            else:
                await query.edit_message_text(
                    "❌ *У вас нет прав для просмотра статистики*",
                    reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                    parse_mode=ParseMode.MARKDOWN
                )
                return MAIN_MENU
        
        elif data == "qr_stats_refresh":
            await qr_stats_command(update, context)
            return SCAN_QR
        
        elif data == "qr_clear_cache":
            cleared = qr_manager.clear_cache(older_than=3600)
            await query.edit_message_text(
                f"🧹 *Кэш QR-кодов очищен*\n\n"
                f"Удалено файлов: {cleared}",
                parse_mode=ParseMode.MARKDOWN
            )
            await asyncio.sleep(2)
            await qr_stats_command(update, context)
            return SCAN_QR
        
        elif data == "scan_continue":
            await query.edit_message_text(
                "📱 *Сканирование QR-кода*\n\n"
                "Отправьте следующий QR-код или введите код вручную:",
                parse_mode=ParseMode.MARKDOWN
            )
            return SCAN_QR
        
        elif data == "scan_back_to_menu":
            role = context.user_data.get('user_role', 'user')
            await query.edit_message_text(
                f"🏠 *Главное меню*\n\n"
                f"Выберите действие:",
                reply_markup=get_main_menu_keyboard(role),
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
        
        else:
            await query.edit_message_text(
                "❌ *Неизвестная команда*",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
    
    except Exception as e:
        logger.error(f"Ошибка в обработчике кнопок: {e}")
        
        try:
            await query.edit_message_text(
                "❌ *Произошла ошибка*",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
        except:
            await query.message.reply_text(
                "❌ *Произошла ошибка*",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
        
        return MAIN_MENU

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    text = update.message.text.strip()
    
    if not rate_limiter.check_limit(user_id):
        remaining = rate_limiter.get_remaining(user_id)
        await update.message.reply_text(
            f"⏰ *Слишком много запросов!*\n\n"
            f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    db.update_user_request(user_id)
    
    try:
        if context.user_data.get('scan_mode', False):
            return await handle_qr_scan(update, context)
        
        if 'in_buy_process' in context.user_data:
            if 'name' not in context.user_data:
                if len(text) < 2:
                    await update.message.reply_text(
                        "❌ *Имя слишком короткое*\n\n"
                        "Введите ваше имя и фамилию (например: Александр Иванов):",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_NAME
                
                if not validate_name(text):
                    await update.message.reply_text(
                        "❌ *Некорректное имя*\n\n"
                        "Введите ваше имя и фамилию (только буквы, пробелы, дефисы):",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_NAME
                
                context.user_data['name'] = sanitize_input(text, 100)
                await update.message.reply_text(
                    "📧 *Введите ваш Email*\n\n"
                    "Например: example@gmail.com",
                    parse_mode=ParseMode.MARKDOWN
                )
                return BUY_EMAIL
                
            elif 'email' not in context.user_data:
                if not is_valid_email(text):
                    await update.message.reply_text(
                        "❌ *Некорректный Email*\n\n"
                        "Введите корректный адрес электронной почты (например: example@gmail.com):",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_EMAIL
                
                context.user_data['email'] = sanitize_input(text, 100)
                
                group_size = context.user_data.get('group_size', 1)
                if group_size == 1:
                    context.user_data['guests'] = [context.user_data['name']]
                    
                    ticket_type = context.user_data.get('ticket_type', 'standard')
                    
                    await update.message.reply_text(
                        format_order_summary(
                            context.user_data['name'],
                            context.user_data['email'],
                            group_size,
                            context.user_data['guests'],
                            ticket_type
                        ),
                        reply_markup=get_confirmation_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_CONFIRM
                else:
                    context.user_data['guest_counter'] = 1
                    await update.message.reply_text(
                        f"👥 *Введите имя гостя #{1}*\n\n"
                        "Например: Мария Смирнова",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_GUESTS
                    
            elif 'guests' in context.user_data and 'guest_counter' in context.user_data:
                group_size = context.user_data.get('group_size', 1)
                guest_counter = context.user_data.get('guest_counter', 1)
                
                if len(text) < 2:
                    await update.message.reply_text(
                        "❌ *Имя слишком короткое*\n\n"
                        f"Введите имя гостя #{guest_counter} заново:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_GUESTS
                
                if not validate_name(text):
                    await update.message.reply_text(
                        "❌ *Некорректное имя*\n\n"
                        f"Введите имя гостя #{guest_counter} (только буквы, пробелы, дефисы):",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_GUESTS
                
                context.user_data['guests'].append(sanitize_input(text, 100))
                
                if guest_counter < group_size:
                    context.user_data['guest_counter'] = guest_counter + 1
                    await update.message.reply_text(
                        f"👥 *Введите имя гостя #{guest_counter + 1}*\n\n"
                        "Например: Алексей Петров",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_GUESTS
                else:
                    ticket_type = context.user_data.get('ticket_type', 'standard')
                    
                    await update.message.reply_text(
                        format_order_summary(
                            context.user_data['name'],
                            context.user_data['email'],
                            group_size,
                            context.user_data['guests'],
                            ticket_type
                        ),
                        reply_markup=get_confirmation_keyboard(),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_CONFIRM
        
        elif 'group_size' not in context.user_data and 'ticket_type' in context.user_data:
            if text.isdigit():
                group_size = int(text)
                if 1 <= group_size <= 100:
                    context.user_data['group_size'] = group_size
                    context.user_data['guests'] = []
                    
                    ticket_type = context.user_data.get('ticket_type', 'standard')
                    
                    await update.message.reply_text(
                        format_price_calculation(group_size, ticket_type) + "\n\n"
                        "👉 *Продолжить покупку?*",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("✅ Да, продолжить", callback_data="buy_continue")],
                            [InlineKeyboardButton("❌ Нет, отмена", callback_data="back_to_menu")]
                        ]),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_TICKET_TYPE
                else:
                    await update.message.reply_text(
                        "❌ *Некорректное количество*\n\n"
                        "Введите число от 1 до 100:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return BUY_TICKET_TYPE
        
        elif 'editing_key' in context.user_data:
            if user_id in ADMIN_IDS:
                editing_key = context.user_data['editing_key']
                editing_name = context.user_data.get('editing_name', 'настройки')
                
                if editing_key == 'event_info_text':
                    if event_settings.update_setting('event_info_text', text):
                        await update.message.reply_text(
                            f"✅ *Текст кнопки 'Событие' успешно обновлен!*\n\n"
                            f"Новый текст сохранен.\n\n"
                            f"Можно проверить, нажав кнопку 'Событие' в главном меню.",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        
                        context.user_data.pop('editing_key', None)
                        context.user_data.pop('editing_name', None)
                        
                        role = get_user_role(user_id)
                        await update.message.reply_text(
                            f"🏠 *Главное меню*\n\n"
                            f"Выберите действие:",
                            reply_markup=get_main_menu_keyboard(role),
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return MAIN_MENU
                    else:
                        await update.message.reply_text(
                            f"❌ *Ошибка при обновлении текста*",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return ADMIN_EDIT_TEXT
                
                elif editing_key == 'price_standard' or editing_key == 'price_group' or editing_key == 'price_vip':
                    if not text.isdigit():
                        await update.message.reply_text(
                            f"❌ *Некорректная цена*\n\n"
                            f"Введите цену цифрами (например: 1000):",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return ADMIN_EDIT_TEXT
                    value = int(text)
                    if value <= 0:
                        await update.message.reply_text(
                            f"❌ *Цена должна быть положительным числом*\n\n"
                            f"Введите корректную цену:",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return ADMIN_EDIT_TEXT
                
                elif editing_key == 'group_threshold':
                    if not text.isdigit():
                        await update.message.reply_text(
                            f"❌ *Некорректное число*\n\n"
                            f"Введите порог цифрами (например: 5):",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return ADMIN_EDIT_TEXT
                    value = int(text)
                    if value < 2:
                        await update.message.reply_text(
                            f"❌ *Порог должен быть не менее 2*\n\n"
                            f"Введите корректное значение:",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        return ADMIN_EDIT_TEXT
                
                elif editing_key == 'contact_telegram':
                    value = text
                    if not (value.startswith('@') or value.startswith('https://t.me/')):
                        value = f"@{value.lstrip('@')}"
                
                else:
                    value = text
                
                if event_settings.update_setting(editing_key, value):
                    await update.message.reply_text(
                        f"✅ *{editing_name} успешно обновлена!*\n\n"
                        f"Новое значение: *{value}*",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                    context.user_data.pop('editing_key', None)
                    context.user_data.pop('editing_name', None)
                    
                    role = get_user_role(user_id)
                    await update.message.reply_text(
                        f"🏠 *Главное меню*\n\n"
                        f"Выберите действие:",
                        reply_markup=get_main_menu_keyboard(role),
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return MAIN_MENU
                else:
                    await update.message.reply_text(
                        f"❌ *Ошибка при обновлении {editing_name}*",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_EDIT_TEXT
        
        elif context.user_data.get('creating_promo', False):
            promo_step = context.user_data.get('promo_step', 'code')
            
            if promo_step == 'code':
                if not re.match(r'^[A-Za-z0-9]+$', text):
                    await update.message.reply_text(
                        "❌ *Некорректный код промокода*\n\n"
                        "Используйте только латинские буквы и цифры.\n"
                        "Попробуйте еще раз:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                existing_promo = db.get_promo_code(text.upper())
                if existing_promo:
                    await update.message.reply_text(
                        f"❌ *Промокод {text.upper()} уже существует!*\n\n"
                        "Введите другой код:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                context.user_data['promo_code'] = text.upper()
                context.user_data['promo_step'] = 'type'
                
                await update.message.reply_text(
                    "🎫 *Создание промокода*\n\n"
                    "Шаг 2/4: Выберите тип скидки:\n\n"
                    "1. Процентная скидка (например, 10%)\n"
                    "2. Фиксированная скидка (например, 100₽)\n\n"
                    "Введите '1' для процентной или '2' для фиксированной:",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_CREATE_PROMO
            
            elif promo_step == 'type':
                if text not in ['1', '2']:
                    await update.message.reply_text(
                        "❌ *Некорректный выбор*\n\n"
                        "Введите '1' для процентной или '2' для фиксированной скидки:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                if text == '1':
                    context.user_data['promo_discount_type'] = 'percent'
                    discount_type_text = "процентную"
                else:
                    context.user_data['promo_discount_type'] = 'fixed'
                    discount_type_text = "фиксированную"
                
                context.user_data['promo_step'] = 'value'
                
                await update.message.reply_text(
                    f"🎫 *Создание промокода*\n\n"
                    f"Шаг 3/4: Введите размер {discount_type_text} скидки:\n\n"
                    f"Пример для процентной: 10 (это 10%)\n"
                    f"Пример для фиксированной: 100 (это 100₽)",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_CREATE_PROMO
            
            elif promo_step == 'value':
                if not text.isdigit():
                    await update.message.reply_text(
                        "❌ *Некорректное значение*\n\n"
                        "Введите число:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                value = int(text)
                discount_type = context.user_data['promo_discount_type']
                
                if discount_type == 'percent' and (value <= 0 or value > 100):
                    await update.message.reply_text(
                        "❌ *Некорректный процент*\n\n"
                        "Процент должен быть от 1 до 100.\n"
                        "Попробуйте еще раз:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                if discount_type == 'fixed' and value <= 0:
                    await update.message.reply_text(
                        "❌ *Некорректная сумма*\n\n"
                        "Сумма должна быть больше 0.\n"
                        "Попробуйте еще раз:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                context.user_data['promo_discount_value'] = value
                context.user_data['promo_step'] = 'max_uses'
                
                await update.message.reply_text(
                    "🎫 *Создание промокода*\n\n"
                    "Шаг 4/4: Введите максимальное количество использований:\n\n"
                    "• Введите число (например, 100)\n"
                    "• Или введите '0' для неограниченного использования",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_CREATE_PROMO
            
            elif promo_step == 'max_uses':
                if not text.isdigit():
                    await update.message.reply_text(
                        "❌ *Некорректное значение*\n\n"
                        "Введите число:",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return ADMIN_CREATE_PROMO
                
                max_uses = int(text)
                if max_uses < 0:
                    max_uses = 1
                
                promo_code = context.user_data['promo_code']
                discount_type = context.user_data['promo_discount_type']
                discount_value = context.user_data['promo_discount_value']
                created_by = update.effective_user.username or f"user_{user_id}"
                
                success = db.create_promo_code(
                    code=promo_code,
                    discount_type=discount_type,
                    discount_value=discount_value,
                    max_uses=max_uses if max_uses > 0 else None,
                    valid_until=None,
                    created_by=created_by
                )
                
                if success:
                    if discount_type == 'percent':
                        discount_text = f"{discount_value}%"
                    else:
                        discount_text = f"{discount_value}₽"
                    
                    max_uses_text = f"{max_uses} раз" if max_uses > 0 else "неограниченно"
                    
                    await update.message.reply_text(
                        f"✅ *Промокод успешно создан!*\n\n"
                        f"*Код:* {promo_code}\n"
                        f"*Тип скидки:* {discount_text}\n"
                        f"*Макс. использований:* {max_uses_text}\n"
                        f"*Создал:* @{created_by}\n\n"
                        f"Промокод активен и готов к использованию!",
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await update.message.reply_text(
                        "❌ *Ошибка при создании промокода*",
                        parse_mode=ParseMode.MARKDOWN
                    )
                
                context.user_data.pop('creating_promo', None)
                context.user_data.pop('promo_step', None)
                context.user_data.pop('promo_code', None)
                context.user_data.pop('promo_discount_type', None)
                context.user_data.pop('promo_discount_value', None)
                
                return await promo_manage_command(update, context)
        
        elif context.user_data.get('viewing_promo', False):
            promo_code = text.upper()
            promo = db.get_promo_code(promo_code)
            
            if not promo:
                await update.message.reply_text(
                    f"❌ *Промокод {promo_code} не найден*",
                    parse_mode=ParseMode.MARKDOWN
                )
                return ADMIN_VIEW_PROMO
            
            status = "🟢 Активен" if promo['is_active'] else "🔴 Неактивен"
            
            if promo['discount_type'] == 'percent':
                discount_text = f"{promo['discount_value']}%"
            else:
                discount_text = f"{promo['discount_value']}₽"
            
            max_uses = promo['max_uses'] or "∞"
            used_count = promo['used_count']
            
            if max_uses != "∞":
                usage_percent = int((used_count / max_uses) * 100)
                usage_text = f"{used_count}/{max_uses} ({usage_percent}%)"
            else:
                usage_text = f"{used_count}/∞"
            
            valid_until = promo['valid_until']
            if valid_until:
                if isinstance(valid_until, str):
                    valid_date = valid_until[:10]
                else:
                    valid_date = valid_until.strftime('%Y-%m-%d')
                valid_text = f"до {valid_date}"
            else:
                valid_text = "без ограничения"
            
            created_at = promo['created_at']
            if isinstance(created_at, str):
                created_date = created_at[:10]
            else:
                created_date = created_at.strftime('%Y-%m-%d')
            
            text = (
                f"🎫 *Информация о промокоде*\n\n"
                f"*Код:* {promo['code']}\n"
                f"*Статус:* {status}\n"
                f"*Тип скидки:* {discount_text}\n"
                f"*Использовано:* {usage_text}\n"
                f"*Действует:* {valid_text}\n"
                f"*Создан:* {created_date}\n"
                f"*Создал:* {promo.get('created_by', 'система')}"
            )
            
            keyboard = [
                [InlineKeyboardButton("🔙 Назад к списку", callback_data="admin_view_promo_list")]
            ]
            
            await update.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
            
            context.user_data.pop('viewing_promo', None)
            return ADMIN_VIEW_PROMO
        
        else:
            role = context.user_data.get('user_role', 'user')
            await update.message.reply_text(
                f"🏠 *Главное меню*\n\n"
                f"Выберите действие:",
                reply_markup=get_main_menu_keyboard(role),
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
    
    except Exception as e:
        logger.error(f"Ошибка в обработчике текста: {e}")
        
        await update.message.reply_text(
            "❌ *Произошла ошибка*\n\n"
            "Пожалуйста, попробуйте еще раз.",
            parse_mode=ParseMode.MARKDOWN
        )
        
        role = get_user_role(user_id)
        return MAIN_MENU

async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS + PROMOTER_IDS:
        if update.message:
            await update.message.reply_text(
                "❌ *У вас нет прав для просмотра панели управления*",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
        elif update.callback_query:
            await update.callback_query.answer()
            await update.callback_query.edit_message_text(
                "❌ *У вас нет прав для просмотра панели управления*",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
        return MAIN_MENU
    
    try:
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            message = query.message
            edit_message = query.edit_message_text
        else:
            message = update.message
            edit_message = update.message.reply_text
        
        await edit_message(
            "📊 *Загружаю данные для панели управления...*",
            parse_mode=ParseMode.MARKDOWN
        )
        
        stats = db.get_statistics()
        scan_stats = db.get_scan_stats() if user.id in SCANNER_IDS else {}
        
        text = "📈 *ПАНЕЛЬ УПРАВЛЕНИЯ SMILE PARTY*\n\n"
        
        text += "📊 *ОСНОВНАЯ СТАТИСТИКА:*\n"
        text += f"• Всего заказов: {stats.get('total_orders', 0)}\n"
        text += f"• Активные: {stats.get('active_orders', 0)}\n"
        text += f"• Закрытые: {stats.get('closed_orders', 0)}\n"
        text += f"• Выручка: {stats.get('revenue', 0)} ₽\n"
        text += f"• Гостей в списках: {stats.get('total_guests', 0)}\n\n"
        
        text += "📅 *СЕГОДНЯ:*\n"
        text += f"• Новых заказов: {stats.get('today_orders', 0)}\n"
        text += f"• Выручка: {stats.get('today_revenue', 0)} ₽\n"
        text += f"• Уникальных покупателей: {stats.get('today_users', 0)}\n\n"
        
        text += "🎫 *СТАТИСТИКА ПО БИЛЕТАМ:*\n"
        text += f"• Обычные: {stats.get('standard_tickets', 0)} ({stats.get('standard_revenue', 0)} ₽)\n"
        text += f"• VIP: {stats.get('vip_tickets', 0)} ({stats.get('vip_revenue', 0)} ₽)\n\n"
        
        if scan_stats:
            text += "📱 *СТАТИСТИКА СКАНИРОВАНИЙ:*\n"
            text += f"• Всего сканирований: {scan_stats.get('total_scans', 0)}\n"
            text += f"• ✅ Успешных: {scan_stats.get('success_scans', 0)}\n"
            text += f"• ⚠️ Повторных: {scan_stats.get('warning_scans', 0)}\n"
            text += f"• ❌ Ошибок: {scan_stats.get('error_scans', 0)}\n"
            text += f"• Отсканировано билетов: {scan_stats.get('scanned_tickets', 0)}/{scan_stats.get('total_valid_tickets', 0)}\n"
            text += f"• Сегодня: {scan_stats.get('today_scans', 0)} (успешно: {scan_stats.get('today_success', 0)})\n\n"
            
            if scan_stats.get('recent_scans'):
                text += "📋 *ПОСЛЕДНИЕ СКАНИРОВАНИЯ:*\n"
                for scan in scan_stats['recent_scans'][:5]:
                    created_at = scan['created_at']
                    if isinstance(created_at, str):
                        time_str = created_at[11:16]
                    else:
                        time_str = created_at.strftime('%H:%M')
                    
                    emoji = "✅" if scan['scan_result'] == 'success' else "⚠️" if scan['scan_result'] == 'warning' else "❌"
                    text += f"{emoji} {time_str} - @{scan['scanner_username']} - {scan['order_code']}\n"
                text += "\n"
        
        weekly_stats = stats.get('weekly_stats', [])
        if weekly_stats:
            text += "📆 *СТАТИСТИКА ЗА 7 ДНЕЙ:*\n"
            
            max_orders = max([day['orders'] for day in weekly_stats] + [1])
            
            for day in weekly_stats[-7:]:
                date_str = day['date']
                if isinstance(date_str, str):
                    date_display = date_str[-5:]
                else:
                    date_display = date_str.strftime('%d.%m')
                
                orders = day['orders']
                revenue = day['revenue'] or 0
                
                bar_length = int((orders / max_orders) * 20)
                bar = '█' * bar_length + '░' * (20 - bar_length)
                
                text += f"{date_display}: {bar} {orders} зак. ({revenue} ₽)\n"
            
            text += "\n"
        
        top_promoters = stats.get('top_promoters', [])
        if top_promoters:
            text += "🏆 *ТОП ПРОМОУТЕРОВ:*\n"
            for i, promoter in enumerate(top_promoters[:5], 1):
                text += f"{i}. @{promoter['username']}: {promoter['closed_count']} зак. ({promoter['total_revenue']} ₽)\n"
            text += "\n"
        
        if scan_stats and scan_stats.get('top_scanners'):
            text += "📱 *ТОП СКАНЕРОВ:*\n"
            for i, scanner in enumerate(scan_stats['top_scanners'][:3], 1):
                text += f"{i}. @{scanner['scanner_username']}: {scanner['scan_count']} сканирований\n"
            text += "\n"
        
        top_users = db.get_top_users(5)
        if top_users:
            text += "👥 *САМЫЕ АКТИВНЫЕ ПОЛЬЗОВАТЕЛИ:*\n"
            for i, user_data in enumerate(top_users, 1):
                username = user_data.get('username', f"user_{user_data['user_id']}")
                first_name = user_data.get('first_name', '')
                request_count = user_data.get('request_count', 0)
                text += f"{i}. {first_name} (@{username}): {request_count} запросов\n"
        
        keyboard = []
        if user.id in ADMIN_IDS:
            keyboard.append([
                InlineKeyboardButton("📤 Экспорт данных", callback_data="admin_export"),
                InlineKeyboardButton("💾 Создать бэкап", callback_data="admin_backup")
            ])
            keyboard.append([
                InlineKeyboardButton("📢 Создать рассылку", callback_data="admin_broadcast"),
                InlineKeyboardButton("🎫 Управление промокодами", callback_data="admin_promo_codes")
            ])
            keyboard.append([
                InlineKeyboardButton("📊 Расширенная статистика QR", callback_data="qr_stats_refresh")
            ])
        
        if user.id in SCANNER_IDS:
            keyboard.append([
                InlineKeyboardButton("📱 Сканировать QR-код", callback_data="scan_qr_menu")
            ])
        
        keyboard.append([
            InlineKeyboardButton("🔄 Обновить", callback_data="admin_dashboard_refresh"),
            InlineKeyboardButton("🔙 Назад", callback_data="admin_back")
        ])
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.MARKDOWN
            )
        
        return ADMIN_DASHBOARD
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке панели управления: {e}")
        
        error_text = f"❌ *Ошибка загрузки панели управления:*\n\n{str(e)}"
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                error_text,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                error_text,
                parse_mode=ParseMode.MARKDOWN
            )
        
        return MAIN_MENU

async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS:
        await update.message.reply_text(
            "❌ *У вас нет прав администратора*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    try:
        await update.message.reply_text(
            "📊 *Подготавливаю данные для экспорта...*",
            parse_mode=ParseMode.MARKDOWN
        )
        
        orders = db.get_orders_by_status("closed")
        
        if not orders:
            await update.message.reply_text(
                "❌ *Нет данных для экспорта*",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        output = io.StringIO()
        writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow([
            'ID заказа', 'Код заказа', 'Тип билета', 'Имя', 'Email', 
            'Telegram', 'Кол-во гостей', 'Сумма', 'Дата создания', 
            'Дата закрытия', 'Промоутер', 'Статус', 'Отсканирован', 'Сканирован', 'Версия QR'
        ])
        
        for order in orders:
            created_at = order['created_at']
            if isinstance(created_at, str):
                created_date = created_at[:10]
            else:
                created_date = created_at.strftime('%Y-%m-%d') if created_at else ''
            
            closed_at = order.get('closed_at')
            if closed_at:
                if isinstance(closed_at, str):
                    closed_date = closed_at[:10]
                else:
                    closed_date = closed_at.strftime('%Y-%m-%d') if closed_at else ''
            else:
                closed_date = ''
            
            scanned = 'Да' if order.get('scanned_at') else 'Нет'
            scanned_by = order.get('scanned_by', '')
            qr_version = order.get('qr_version', '')
            
            writer.writerow([
                order['order_id'],
                order['order_code'],
                'VIP' if order.get('ticket_type') == 'vip' else 'Standard',
                sanitize_input(order['user_name']),
                sanitize_input(order['user_email']),
                sanitize_input(order.get('username', '')),
                order['group_size'],
                order['total_amount'],
                created_date,
                closed_date,
                sanitize_input(order.get('closed_by', '')),
                order['status'],
                scanned,
                scanned_by,
                qr_version
            ])
        
        output.seek(0)
        csv_data = output.getvalue().encode('utf-8-sig')
        
        await update.message.reply_document(
            document=io.BytesIO(csv_data),
            filename=f"orders_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            caption="📊 Экспорт заказов"
        )
        
        logger.info(f"Экспорт данных выполнен, отправлено {len(orders)} записей")
        
    except Exception as e:
        logger.error(f"Ошибка экспорта данных: {e}")
        await update.message.reply_text(
            f"❌ *Ошибка экспорта данных:*\n\n{str(e)}",
            parse_mode=ParseMode.MARKDOWN
        )

async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS:
        await update.message.reply_text(
            "❌ *У вас нет прав администратора*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    backup_sql = f"{backup_file}.sql"
    
    try:
        shutil.copy2(DB_FILE, backup_file)
        
        with closing(sqlite3.connect(DB_FILE)) as conn:
            with open(backup_sql, 'w', encoding='utf-8') as f:
                for line in conn.iterdump():
                    f.write(f'{line}\n')
        
        with open(backup_file, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=backup_file,
                caption="💾 Резервная копия базы данных"
            )
        
        with open(backup_sql, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=backup_sql,
                caption="📝 SQL дамп базы данных"
            )
        
        os.remove(backup_file)
        os.remove(backup_sql)
        
        await update.message.reply_text(
            "✅ *Резервные копии успешно созданы и отправлены!*",
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Ошибка создания бэкапа: {e}")
        await update.message.reply_text(
            f"❌ *Ошибка создания резервной копии:*\n\n{str(e)}",
            parse_mode=ParseMode.MARKDOWN
        )

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS:
        await update.message.reply_text(
            "❌ *У вас нет прав администратора*",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    if context.args:
        message = ' '.join(context.args)
        
        users = db.get_all_users()
        
        await update.message.reply_text(
            f"📢 *Начинаю рассылку для {len(users)} пользователей...*",
            parse_mode=ParseMode.MARKDOWN
        )
        
        success = 0
        failed = 0
        
        for user_data in users:
            try:
                await context.bot.send_message(
                    chat_id=user_data['user_id'],
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
                success += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                failed += 1
                logger.error(f"Ошибка отправки пользователю {user_data['user_id']}: {e}")
        
        await update.message.reply_text(
            f"✅ *Рассылка завершена!*\n\n"
            f"✅ Успешно: {success}\n"
            f"❌ Не удалось: {failed}",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "📢 *Создание рассылки*\n\n"
            "Введите сообщение для рассылки:\n\n"
            "Пример: /broadcast Привет! Скоро начнется SMILE PARTY! 🎉",
            parse_mode=ParseMode.MARKDOWN
        )

async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user = update.effective_user
        
        if not rate_limiter.check_limit(user.id):
            remaining = rate_limiter.get_remaining(user.id)
            await update.message.reply_text(
                f"⏰ *Слишком много запросов!*\n\n"
                f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
                parse_mode=ParseMode.MARKDOWN
            )
            return MAIN_MENU
        
        if user.id in ADMIN_IDS:
            await update.message.reply_text(
                "📋 *Собираю логи...*",
                parse_mode=ParseMode.MARKDOWN
            )
            
            stats = db.get_statistics()
            scan_stats = db.get_scan_stats()
            qr_stats = db.get_qr_statistics()
            
            recent_orders = []
            try:
                with closing(db.get_connection()) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT 10")
                    recent_orders = [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Ошибка получения последних заказов: {e}")
            
            log_message = (
                "📊 *ЛОГИ БОТА*\n\n"
                f"*📅 Время:* {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
                f"*👤 Администратор:* {user.username if user.username else user.id}\n\n"
                f"*📈 СТАТИСТИКА:*\n"
                f"• Всего заказов: {stats.get('total_orders', 0)}\n"
                f"• Активные: {stats.get('active_orders', 0)}\n"
                f"• Закрытые: {stats.get('closed_orders', 0)}\n"
                f"• Отложенные: {stats.get('deferred_orders', 0)}\n"
                f"• Возвраты: {stats.get('refunded_orders', 0)}\n"
                f"• Выручка: {stats.get('revenue', 0)} ₽\n"
                f"• Всего гостей: {stats.get('total_guests', 0)}\n\n"
                f"*📱 СТАТИСТИКА СКАНИРОВАНИЙ:*\n"
                f"• Всего сканирований: {scan_stats.get('total_scans', 0)}\n"
                f"• Успешных: {scan_stats.get('success_scans', 0)}\n"
                f"• Повторных: {scan_stats.get('warning_scans', 0)}\n"
                f"• Отсканировано билетов: {scan_stats.get('scanned_tickets', 0)}/{scan_stats.get('total_valid_tickets', 0)}\n"
                f"• Сегодня: {scan_stats.get('today_scans', 0)} (успешно: {scan_stats.get('today_success', 0)})\n\n"
                f"*💾 СТАТИСТИКА КЭША QR:*\n"
                f"• Попаданий в кэш: {qr_stats.get('cache_hits', 0)}\n"
                f"• Промахов: {qr_stats.get('cache_misses', 0)}\n"
                f"• Эффективность: {qr_stats.get('cache_hit_rate', 0)}%\n"
                f"• Среднее время генерации: {qr_stats.get('avg_generation_time', 0)} мс\n\n"
            )
            
            if recent_orders:
                log_message += "*📋 ПОСЛЕДНИЕ 10 ЗАКАЗОВ:*\n"
                for order in recent_orders:
                    created_at = order['created_at']
                    if isinstance(created_at, str):
                        created_date = created_at[:16].replace('T', ' ')
                    else:
                        created_date = created_at.strftime('%d.%m.%Y %H:%M')
                    
                    scanned = "✅" if order.get('scanned_at') else "⏳"
                    
                    log_message += (
                        f"• #{order['order_id']} | {order['status']} | {scanned} | "
                        f"{order['group_size']} чел. | {order['total_amount']} ₽ | "
                        f"{created_date}\n"
                    )
            
            await send_log_to_channel(context, f"Логи запрошены администратором {user.username if user.username else user.id}")
            
            await update.message.reply_text(
                log_message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            await update.message.reply_text(
                "✅ *Логи отправлены в канал и отображены выше*",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "❌ *У вас нет прав администратора*",
                parse_mode=ParseMode.MARKDOWN
            )
    except Exception as e:
        logger.error(f"Ошибка в команде logs: {e}")
        await update.message.reply_text(
            "❌ *Произошла ошибка при получении логов*",
            parse_mode=ParseMode.MARKDOWN
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    
    if not rate_limiter.check_limit(user.id):
        remaining = rate_limiter.get_remaining(user.id)
        await update.message.reply_text(
            f"⏰ *Слишком много запросов!*\n\n"
            f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    help_text = (
        "🎉 *SMILE PARTY Бот - Помощь*\n\n"
        "*Основные команды:*\n"
        "• /start - Начать работу с ботом\n"
        "• /help - Показать это сообщение\n"
        "• /cancel - Отменить текущее действие\n"
        "• /logs - Получить логи (только для администраторов)\n"
        "• /export - Экспорт данных в CSV (админы)\n"
        "• /backup - Создать резервную копию (админы)\n"
        "• /broadcast <текст> - Рассылка сообщений (админы)\n"
        "• /dashboard - Панель управления (админы/промоутеры)\n"
        "• /scanqr - Сканировать QR-код (только для промоутеров и администраторов)\n"
        "• /scanstats - Статистика сканирований (админы/промоутеры)\n"
        "• /qrstats - Расширенная статистика QR (админы)\n\n"
        "*Функции для всех:*\n"
        "• Узнать цены на билеты\n"
        "• Купить билеты онлайн\n"
        "• Просмотреть информацию о мероприятии\n"
        "• Посмотреть свои заказы\n"
        "• Получить QR-коды для своих билетов\n\n"
        "*Для промоутеров и администраторов:*\n"
        "• Просмотр активных заявок\n"
        "• Обработка заказов\n"
        "• Отслеживание статистики\n"
        "• Панель управления\n"
        "• Сканирование QR-кодов на входе (с защитой от повторного использования)\n\n"
        "*Для администраторов:*\n"
        "• Управление настройками\n"
        "• Просмотр статистики\n"
        "• Редактирование информации о мероприятии\n"
        "• Получение логов\n"
        "• Управление промокодами\n"
        "• Экспорт данных\n"
        "• Резервное копирование\n"
        "• Рассылка сообщений\n"
        "• Мониторинг производительности QR-системы\n\n"
        "*Защита QR-кодов:*\n"
        "• HMAC подпись - защита от подделки\n"
        "• Timestamp - ограничение срока действия\n"
        "• Rate limiting - защита от повторного сканирования\n"
        "• Хэширование гостей - приватность данных\n"
        "• Кэширование - быстрая генерация\n\n"
        "*Техническая поддержка:* @smile_party"
    )
    
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN
    )

async def notify_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if not rate_limiter.check_limit(user.id):
        remaining = rate_limiter.get_remaining(user.id)
        await update.message.reply_text(
            f"⏰ *Слишком много запросов!*\n\n"
            f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if user.id in ADMIN_IDS:
        await update.message.reply_text(
            "🔄 *Начинаю отправку уведомлений о перезапуске...*",
            parse_mode=ParseMode.MARKDOWN
        )
        
        import threading
        thread = threading.Thread(target=send_restart_notifications)
        thread.start()
        
        await update.message.reply_text(
            "✅ *Запущена отправка уведомлений всем пользователям*\n\n"
            "Уведомления отправляются в фоновом режиме. Проверьте логи для деталей.",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "❌ *У вас нет прав администратора*",
            parse_mode=ParseMode.MARKDOWN
        )

async def check_new_orders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if not rate_limiter.check_limit(user.id):
        remaining = rate_limiter.get_remaining(user.id)
        await update.message.reply_text(
            f"⏰ *Слишком много запросов!*\n\n"
            f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if user.id in ADMIN_IDS or user.id in PROMOTER_IDS:
        await update.message.reply_text(
            "🔄 *Проверяю новые заказы...*",
            parse_mode=ParseMode.MARKDOWN
        )
        
        unnotified_orders = db.get_unnotified_orders()
        
        if unnotified_orders:
            await update.message.reply_text(
                f"✅ *Найдено {len(unnotified_orders)} новых заказов*\n\n"
                "Отправляю уведомления...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            for order in unnotified_orders:
                await send_new_order_notification(context, order)
                await asyncio.sleep(1)
            
            await update.message.reply_text(
                f"✅ *Уведомления отправлены по {len(unnotified_orders)} заказам*",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "✅ *Нет новых заказов для уведомления*",
                parse_mode=ParseMode.MARKDOWN
            )
    else:
        await update.message.reply_text(
            "❌ *У вас нет прав для этой команды*",
            parse_mode=ParseMode.MARKDOWN
        )

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    
    if not rate_limiter.check_limit(user.id):
        remaining = rate_limiter.get_remaining(user.id)
        await update.message.reply_text(
            f"⏰ *Слишком много запросов!*\n\n"
            f"Пожалуйста, подождите. Доступно запросов через 60 секунд: {remaining}",
            parse_mode=ParseMode.MARKDOWN
        )
        return MAIN_MENU
    
    context.user_data.pop('in_buy_process', None)
    context.user_data.pop('name', None)
    context.user_data.pop('email', None)
    context.user_data.pop('group_size', None)
    context.user_data.pop('guests', None)
    context.user_data.pop('guest_counter', None)
    context.user_data.pop('editing_key', None)
    context.user_data.pop('editing_name', None)
    context.user_data.pop('ticket_type', None)
    context.user_data.pop('creating_promo', None)
    context.user_data.pop('promo_step', None)
    context.user_data.pop('promo_code', None)
    context.user_data.pop('promo_discount_type', None)
    context.user_data.pop('promo_discount_value', None)
    context.user_data.pop('viewing_promo', None)
    context.user_data.pop('scan_mode', None)
    
    await update.message.reply_text(
        "❌ *Действие отменено*",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.MARKDOWN
    )
    
    role = get_user_role(user.id)
    context.user_data['user_role'] = role
    
    await update.message.reply_text(
        f"🏠 *Главное меню*\n\n"
        f"Выберите действие:",
        reply_markup=get_main_menu_keyboard(role),
        parse_mode=ParseMode.MARKDOWN
    )
    
    return MAIN_MENU

async def promo_manage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    if user.id not in ADMIN_IDS:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                "❌ *У вас нет прав администратора*",
                reply_markup=get_main_menu_keyboard(context.user_data.get('user_role', 'user')),
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "❌ *У вас нет прав администратора*",
                parse_mode=ParseMode.MARKDOWN
            )
        return MAIN_MENU
    
    promos = db.get_all_promo_codes()
    
    if not promos:
        text = "🎫 *Управление промокодами*\n\n"
        text += "У вас пока нет созданных промокодов.\n\n"
        text += "Нажмите 'Создать промокод', чтобы добавить новый."
    else:
        text = "🎫 *Управление промокодами*\n\n"
        text += f"Всего промокодов: {len(promos)}\n\n"
        
        for promo in promos[:10]:
            status = "🟢" if promo['is_active'] else "🔴"
            
            if promo['discount_type'] == 'percent':
                discount = f"{promo['discount_value']}%"
            else:
                discount = f"{promo['discount_value']}₽"
            
            max_uses = promo['max_uses'] or "∞"
            used = promo['used_count']
            
            text += f"{status} `{promo['code']}` | {discount} | Использовано: {used}/{max_uses}\n"
        
        if len(promos) > 10:
            text += f"\n...и еще {len(promos) - 10} промокодов"
    
    keyboard = [
        [InlineKeyboardButton("➕ Создать промокод", callback_data="admin_create_promo")],
        [InlineKeyboardButton("🔍 Найти промокод", callback_data="admin_view_promo")],
        [InlineKeyboardButton("🔙 Назад", callback_data="admin_menu")]
    ]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    return ADMIN_MENU

async def periodic_notification_check(context: ContextTypes.DEFAULT_TYPE):
    await check_and_send_notifications(context)
    await send_reminders(context)

async def send_restart_notifications_async(bot_token: str):
    try:
        from telegram import Bot
        
        bot = Bot(token=bot_token)
        users = db.get_users_to_notify()
        settings_data = event_settings.get_all_settings()
        
        notification_count = 0
        for user in users:
            try:
                await bot.send_message(
                    chat_id=user['user_id'],
                    text=f"🔄 *{escape_markdown(str(settings_data['event_name']))} бот перезапущен!*\n\n"
                         f"Бот снова в сети и готов к работе.\n"
                         f"Теперь с улучшенной системой QR-кодов с защитой от подделок!\n"
                         f"Используйте /start для начала работы.",
                    parse_mode=ParseMode.MARKDOWN
                )
                db.mark_user_notified(user['user_id'])
                notification_count += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Не удалось отправить уведомление пользователю {user['user_id']}: {e}")
        
        logger.info(f"✅ Отправлено {notification_count} уведомлений при перезапуске бота")
        
    except Exception as e:
        logger.error(f"Ошибка при отправке уведомлений при перезапуске: {e}")

def send_restart_notifications():
    import asyncio
    asyncio.run(send_restart_notifications_async(BOT_TOKEN))

def main() -> None:
    logger.info("🚀 Запуск SMILE PARTY Bot с ULTIMATE QR SYSTEM...")
    logger.info(f"👥 Права на сканирование QR-кодов имеют {len(SCANNER_IDS)} пользователей")
    logger.info(f"🔒 Защита QR-кодов: HMAC + Timestamp + Версионирование")
    logger.info(f"📱 Версия QR-формата: {QR_CONFIG['version']}")
    logger.info(f"💾 Кэширование QR-кодов: {'Включено' if QR_CONFIG['enable_qr_caching'] else 'Выключено'}")
    logger.info(f"📊 Мониторинг производительности: Включен")
    
    if CV2_AVAILABLE:
        logger.info("✅ OpenCV доступен для распознавания QR-кодов")
    else:
        logger.warning("⚠️ OpenCV не установлен. Для улучшенного распознавания: pip install opencv-python")
    
    db.reset_notification_status()
    
    application = ApplicationBuilder().token(BOT_TOKEN).concurrent_updates(True).build()
    
    try:
        job_queue = application.job_queue
        if job_queue:
            job_queue.run_repeating(periodic_notification_check, interval=30, first=10)
            job_queue.run_repeating(send_reminders, interval=1800, first=300)
            job_queue.run_once(lambda _: qr_manager.clear_cache(86400), when=3600)
            job_queue.run_daily(lambda _: qr_manager.clear_cache(86400), time=datetime.time(hour=3, minute=0))
            
            logger.info("✅ Запущены периодические задачи")
        else:
            logger.warning("⚠️ JobQueue не доступен. Для периодических задач установите: pip install 'python-telegram-bot[job-queue]'")
    except Exception as e:
        logger.warning(f"⚠️ JobQueue не доступен: {e}")
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            ROLE_SELECTION: [CallbackQueryHandler(button_handler)],
            MAIN_MENU: [
                CallbackQueryHandler(button_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            BUY_TICKET_TYPE: [
                CallbackQueryHandler(button_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            BUY_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            BUY_EMAIL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            BUY_GUESTS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            BUY_CONFIRM: [CallbackQueryHandler(button_handler)],
            ADMIN_MENU: [CallbackQueryHandler(button_handler)],
            PROMOTER_MENU: [CallbackQueryHandler(button_handler)],
            ADMIN_EDIT: [CallbackQueryHandler(button_handler)],
            ADMIN_EDIT_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            PROMOTER_VIEW_ORDER: [CallbackQueryHandler(button_handler)],
            PROMOTER_DEFERRED: [CallbackQueryHandler(button_handler)],
            ADMIN_RESET_STATS: [CallbackQueryHandler(button_handler)],
            ADMIN_CREATE_PROMO: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            ADMIN_VIEW_PROMO: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
                CallbackQueryHandler(button_handler)
            ],
            ADMIN_BROADCAST: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
            ],
            ADMIN_DASHBOARD: [CallbackQueryHandler(button_handler)],
            ADMIN_EXPORT_DATA: [CallbackQueryHandler(button_handler)],
            SCAN_QR: [
                MessageHandler(filters.PHOTO, handle_photo),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
                CallbackQueryHandler(button_handler)
            ],
            SCAN_RESULT: [CallbackQueryHandler(button_handler)]
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            CommandHandler("start", start_command),
            CommandHandler("help", help_command),
            CommandHandler("notify_all", notify_all_command),
            CommandHandler("check_orders", check_new_orders_command),
            CommandHandler("logs", logs_command),
            CommandHandler("export", export_command),
            CommandHandler("backup", backup_command),
            CommandHandler("broadcast", broadcast_command),
            CommandHandler("dashboard", dashboard_command),
            CommandHandler("scanqr", scan_qr_command),
            CommandHandler("scanstats", scan_stats_command),
            CommandHandler("qrstats", qr_stats_command)
        ]
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("notify_all", notify_all_command))
    application.add_handler(CommandHandler("check_orders", check_new_orders_command))
    application.add_handler(CommandHandler("logs", logs_command))
    application.add_handler(CommandHandler("export", export_command))
    application.add_handler(CommandHandler("backup", backup_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("dashboard", dashboard_command))
    application.add_handler(CommandHandler("scanqr", scan_qr_command))
    application.add_handler(CommandHandler("scanstats", scan_stats_command))
    application.add_handler(CommandHandler("qrstats", qr_stats_command))
    application.add_handler(CallbackQueryHandler(qr_stats_callback, pattern="^qr_"))
    
    logger.info("✅ Бот с ULTIMATE QR SYSTEM запущен и готов к работе!")
    logger.info(f"📱 Команды QR: /scanqr, /scanstats, /qrstats")
    logger.info(f"🔒 Все QR-коды защищены HMAC подписью и временной меткой")
    
    import threading
    import time
    
    def send_notifications_delayed():
        time.sleep(5)
        logger.info("🔄 Начинаю отправку уведомлений о перезапуске...")
        send_restart_notifications()
    
    notification_thread = threading.Thread(target=send_notifications_delayed)
    notification_thread.daemon = True
    notification_thread.start()
    
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()