"""
DeepInd 日志工具
提供统一的日志格式和输出
"""

import logging
import sys
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取带统一格式的 logger

    Args:
        name: logger 名称，格式为 "DeepInd::模块名"
        level: 日志级别

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台输出
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # 统一格式: [时间戳] [级别] [模块名] 消息
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_section(logger: logging.Logger, title: str, char: str = '='):
    """打印分隔标题"""
    logger.info(char * 60)
    logger.info(title)
    logger.info(char * 60)
