"""
Health check functions for the Streamlit UI
"""

import logging
import time
import requests
import streamlit as st

logger = logging.getLogger("streamlit-ui")


@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid too many requests
def get_system_health(backend_url: str = None):
    """Get comprehensive system health from backend."""
    backend_api_url = backend_url or st.session_state.get(
        "backend_api_url", "http://192.168.0.110:8000"
    )

    logger.info("🏥 Performing system health check")
    start_time = time.time()

    try:
        # First try the simple readiness check with shorter timeout
        logger.debug("🔍 Checking backend readiness...")
        response = requests.get(f"{backend_api_url}/readiness", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend readiness check passed")
            # Backend is responding, now try the full health check with longer timeout
            try:
                logger.debug("🔍 Performing full health check...")
                health_response = requests.get(f"{backend_api_url}/health", timeout=30)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    duration = time.time() - start_time
                    logger.info(f"✅ Full health check completed in {duration:.3f}s")
                    logger.debug(f"Health data: {health_data}")
                    return health_data
                else:
                    # Health check failed but backend is responsive
                    duration = time.time() - start_time
                    logger.warning(
                        f"⚠️ Health check failed with status {health_response.status_code} in {duration:.3f}s"
                    )
                    return {
                        "status": "partial",
                        "overall_healthy": False,
                        "services": {
                            "backend": {
                                "status": f"⚠️ Backend responsive but health check failed: HTTP {health_response.status_code}",
                                "healthy": False,
                            }
                        },
                        "error": "Health check endpoint returned unexpected status code",
                    }
            except requests.exceptions.Timeout:
                # Health check timed out but backend is responsive
                duration = time.time() - start_time
                logger.warning(f"⚠️ Health check timed out in {duration:.3f}s")
                return {
                    "status": "partial",
                    "overall_healthy": False,
                    "services": {
                        "backend": {
                            "status": "⚠️ Backend responsive but health check timed out (some services may be slow)",
                            "healthy": False,
                        }
                    },
                    "error": "Health check timed out - external services may be unavailable",
                }
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Health check error in {duration:.3f}s: {e}")
                return {
                    "status": "partial",
                    "overall_healthy": False,
                    "services": {
                        "backend": {
                            "status": f"⚠️ Backend responsive but health check failed: {str(e)}",
                            "healthy": False,
                        }
                    },
                    "error": str(e),
                }
        else:
            duration = time.time() - start_time
            logger.error(
                f"❌ Backend readiness check failed with status {response.status_code} in {duration:.3f}s"
            )
            return {
                "status": "unhealthy",
                "overall_healthy": False,
                "services": {
                    "backend": {
                        "status": f"❌ Backend API Error: HTTP {response.status_code}",
                        "healthy": False,
                    }
                },
                "error": "Backend API returned unexpected status code",
            }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ System health check failed in {duration:.3f}s: {e}")
        return {
            "status": "unhealthy",
            "overall_healthy": False,
            "services": {
                "backend": {
                    "status": f"❌ Backend API Connection Failed: {str(e)}",
                    "healthy": False,
                }
            },
            "error": str(e),
        }
