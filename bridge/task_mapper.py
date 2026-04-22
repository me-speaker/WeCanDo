"""Task Mapper - 任务类型映射"""
from typing import Dict, List, Any


TASK_TYPE_MAPPING = {
    "ml_pipeline": {
        "modules": ["data_loading", "model_building", "training", "inference"],
        "test_focus": ["data_validation", "model_evaluation"],
        "delivery_includes": ["model_export", "inference_api"]
    },
    "web_api": {
        "modules": ["endpoints", "models", "middleware", "auth"],
        "test_focus": ["endpoint_tests", "integration_tests"],
        "delivery_includes": ["docker", "api_docs"]
    },
    "gui_app": {
        "modules": ["ui", "state_management", "widgets"],
        "test_focus": ["ui_tests", "user_flow_tests"],
        "delivery_includes": ["executable", "installer"]
    },
    "data_pipeline": {
        "modules": ["extract", "transform", "load", "monitoring"],
        "test_focus": ["data_quality", "pipeline_tests"],
        "delivery_includes": ["scheduler", "monitoring_dashboard"]
    }
}


COMPLEXITY_ACTIVATION = {
    "simple": ["ceo", "requirements_analyst", "developer"],
    "medium": ["ceo", "requirements_analyst", "developer", "tester", "delivery"],
    "complex": ["ceo", "requirements_analyst", "architect", "developer", "tester", "delivery"]
}


def get_task_modules(task_type: str) -> List[str]:
    """获取指定任务类型需要的模块列表"""
    return TASK_TYPE_MAPPING.get(task_type, {}).get("modules", [])


def get_test_focus(task_type: str) -> List[str]:
    """获取指定任务类型的测试重点"""
    return TASK_TYPE_MAPPING.get(task_type, {}).get("test_focus", [])


def get_delivery_includes(task_type: str) -> List[str]:
    """获取指定任务类型的交付物列表"""
    return TASK_TYPE_MAPPING.get(task_type, {}).get("delivery_includes", [])


def get_activation_list(complexity: str) -> List[str]:
    """根据复杂度获取需要激活的 Agent 列表"""
    return COMPLEXITY_ACTIVATION.get(complexity, COMPLEXITY_ACTIVATION["simple"])
