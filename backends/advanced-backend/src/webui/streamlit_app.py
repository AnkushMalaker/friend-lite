import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from advanced_omi_backend.debug_system_tracker import get_debug_tracker

load_dotenv()

# Create logs directory for Streamlit app
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure comprehensive logging for Streamlit app
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "streamlit.log")],
)

logger = logging.getLogger("streamlit-ui")
logger.info("🚀 Starting Friend-Lite Streamlit Dashboard")

# ---- Configuration ---- #
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://192.168.0.110:8000")

BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", BACKEND_API_URL)

logger.info(
    f"🔧 Configuration loaded - Backend API: {BACKEND_API_URL}, Public URL: {BACKEND_PUBLIC_URL}"
)


# ---- Authentication Functions ---- #
def init_auth_state():
    """Initialize authentication state in session state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "auth_method" not in st.session_state:
        st.session_state.auth_method = None
    if "auth_config" not in st.session_state:
        st.session_state.auth_config = None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_auth_config():
    """Get authentication configuration from backend."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/api/auth/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get auth config: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error getting auth config: {e}")
        return None


def get_auth_headers():
    """Get authentication headers for API requests."""
    if st.session_state.get("auth_token"):
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}


def check_auth_from_url():
    """Check for authentication token in URL parameters."""
    try:
        # Check URL parameters for token
        query_params = st.query_params
        if "token" in query_params:
            token = query_params["token"]
            logger.info("🔐 Authentication token found in URL parameters")

            # Validate token by calling a protected endpoint
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{BACKEND_API_URL}/api/users", headers=headers, timeout=5)

            if response.status_code == 200:
                st.session_state.authenticated = True
                st.session_state.auth_token = token
                st.session_state.auth_method = "token"

                # Try to get user info from token (decode JWT payload)
                try:
                    import base64

                    # Split JWT token and decode payload
                    token_parts = token.split(".")
                    if len(token_parts) >= 2:
                        # Add padding if needed
                        payload = token_parts[1]
                        payload += "=" * (4 - len(payload) % 4)
                        decoded = base64.b64decode(payload)
                        user_data = json.loads(decoded)
                        st.session_state.user_info = {
                            "user_id": user_data.get("sub", "Unknown"),
                            "email": user_data.get("email", "Unknown"),
                            "name": user_data.get("name", user_data.get("email", "Unknown")),
                        }
                except Exception as e:
                    logger.warning(f"Could not decode user info from token: {e}")
                    st.session_state.user_info = {"user_id": "Unknown", "email": "Unknown"}

                logger.info("✅ Authentication successful from URL token")

                # Clear the token from URL to avoid confusion
                st.query_params.clear()
                st.rerun()
                return True
            else:
                logger.warning("❌ Token validation failed")
                return False

        # Check for error in URL
        if "error" in query_params:
            error = query_params["error"]
            logger.error(f"❌ Authentication error in URL: {error}")
            st.error(f"Authentication error: {error}")
            st.query_params.clear()
            return False

    except Exception as e:
        logger.error(f"❌ Error checking authentication from URL: {e}")
        return False

    return False


def login_with_credentials(email, password):
    """Login with email and password."""
    try:
        logger.info(f"🔐 Attempting login for email: {email}")
        response = requests.post(
            f"{BACKEND_API_URL}/auth/jwt/login",
            data={"username": email, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        if response.status_code == 200:
            auth_data = response.json()
            token = auth_data.get("access_token")

            if token:
                st.session_state.authenticated = True
                st.session_state.auth_token = token
                st.session_state.auth_method = "credentials"
                st.session_state.user_info = {"user_id": email, "email": email, "name": email}
                logger.info("✅ Credential login successful")
                return True, "Login successful!"
            else:
                logger.error("❌ No access token in response")
                return False, "No access token received"
        else:
            error_msg = "Invalid credentials"
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", error_msg)
            except:
                pass
            logger.error(f"❌ Login failed: {error_msg}")
            return False, error_msg

    except requests.exceptions.Timeout:
        logger.error("❌ Login request timed out")
        return False, "Login request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Login request failed: {e}")
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        logger.error(f"❌ Unexpected login error: {e}")
        return False, f"Unexpected error: {str(e)}"


def logout():
    """Logout and clear authentication state."""
    logger.info("🚪 User logging out")
    st.session_state.authenticated = False
    st.session_state.auth_token = None
    st.session_state.user_info = None
    st.session_state.auth_method = None


def generate_jwt_token(email, password):
    """Generate JWT token for given credentials."""
    try:
        logger.info(f"🔑 Generating JWT token for: {email}")
        response = requests.post(
            f"{BACKEND_API_URL}/auth/jwt/login",
            data={"username": email, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        if response.status_code == 200:
            auth_data = response.json()
            token = auth_data.get("access_token")
            token_type = auth_data.get("token_type", "bearer")

            if token:
                logger.info("✅ JWT token generated successfully")
                return True, token, token_type
            else:
                logger.error("❌ No access token in response")
                return False, "No access token received", None
        else:
            error_msg = "Invalid credentials"
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", error_msg)
            except:
                pass
            logger.error(f"❌ Token generation failed: {error_msg}")
            return False, error_msg, None

    except requests.exceptions.Timeout:
        logger.error("❌ Token generation request timed out")
        return False, "Request timed out. Please try again.", None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Token generation request failed: {e}")
        return False, f"Connection error: {str(e)}", None
    except Exception as e:
        logger.error(f"❌ Unexpected token generation error: {e}")
        return False, f"Unexpected error: {str(e)}", None


def show_auth_sidebar():
    """Show authentication status and controls in sidebar."""
    with st.sidebar:
        st.header("🔐 Authentication")

        # Get auth configuration from backend
        auth_config = get_auth_config()

        if st.session_state.get("authenticated", False):
            user_info = st.session_state.get("user_info", {})
            user_name = user_info.get("name", "Unknown User")
            auth_method = st.session_state.get("auth_method", "unknown")

            st.success(f"✅ Logged in as **{user_name}**")
            st.caption(f"Method: {auth_method.title()}")

            # Quick token access for authenticated users
            current_token = st.session_state.get("auth_token")
            if current_token:
                with st.expander("🔑 Your Current Token"):
                    st.text_area(
                        "Current Auth Token:",
                        value=current_token,
                        height=100,
                        help="Your current authentication token",
                        key="current_user_token",
                    )

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(
                            "📋 Copy Current Token",
                            key="copy_current_token",
                            use_container_width=True,
                        ):
                            copy_current_js = f"""
                            <script>
                            function copyCurrentToClipboard() {{
                                const text = `{current_token}`;
                                navigator.clipboard.writeText(text).then(function() {{
                                    console.log('Current token copied to clipboard');
                                }}).catch(function(err) {{
                                    console.error('Could not copy current token: ', err);
                                }});
                            }}
                            copyCurrentToClipboard();
                            </script>
                            """
                            st.components.v1.html(copy_current_js, height=0)
                            st.success("✅ Current token copied!")

                    with col2:
                        if st.button(
                            "📋 Copy Auth Header", key="copy_current_auth", use_container_width=True
                        ):
                            auth_header_current = f"Authorization: Bearer {current_token}"
                            copy_auth_current_js = f"""
                            <script>
                            function copyCurrentAuthToClipboard() {{
                                const text = `{auth_header_current}`;
                                navigator.clipboard.writeText(text).then(function() {{
                                    console.log('Current auth header copied to clipboard');
                                }}).catch(function(err) {{
                                    console.error('Could not copy current auth header: ', err);
                                }});
                            }}
                            copyCurrentAuthToClipboard();
                            </script>
                            """
                            st.components.v1.html(copy_auth_current_js, height=0)
                            st.success("✅ Auth header copied!")

                    st.caption("💡 Use this token for WebSocket connections and API calls")

            if st.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()
        else:
            st.warning("🔒 Not authenticated")

            # Manual token input
            with st.expander("🔑 Manual Token Entry"):
                manual_token = st.text_input(
                    "JWT Token:", type="password", help="Paste token from generated JWT"
                )
                if st.button("Submit Token"):
                    if manual_token.strip():
                        # Validate token
                        headers = {"Authorization": f"Bearer {manual_token.strip()}"}
                        try:
                            response = requests.get(
                                f"{BACKEND_API_URL}/api/users", headers=headers, timeout=5
                            )
                            if response.status_code == 200:
                                st.session_state.authenticated = True
                                st.session_state.auth_token = manual_token.strip()
                                st.session_state.auth_method = "manual"
                                st.session_state.user_info = {
                                    "user_id": "Unknown",
                                    "email": "Unknown",
                                    "name": "Manual Login",
                                }
                                st.success("✅ Token validated successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid token")
                        except Exception as e:
                            st.error(f"❌ Error validating token: {e}")
                    else:
                        st.error("Please enter a token")

            # Email/Password login
            with st.expander("🔑 Email & Password Login", expanded=True):
                with st.form("login_form"):
                    email = st.text_input("Email:")
                    password = st.text_input("Password:", type="password")
                    login_submitted = st.form_submit_button("🔑 Login")

                    if login_submitted:
                        if email.strip() and password.strip():
                            with st.spinner("Logging in..."):
                                success, message = login_with_credentials(
                                    email.strip(), password.strip()
                                )
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        else:
                            st.error("Please enter both email and password")

            # JWT Token Generator
            with st.expander("🔑 Generate JWT Token"):
                st.info("Generate JWT tokens for API access or WebSocket connections")
                with st.form("jwt_token_form"):
                    jwt_email = st.text_input("Email:", placeholder="admin@example.com")
                    jwt_password = st.text_input(
                        "Password:", type="password", placeholder="Admin password"
                    )
                    generate_submitted = st.form_submit_button("🔑 Generate Token")

                    if generate_submitted:
                        if jwt_email.strip() and jwt_password.strip():
                            with st.spinner("Generating JWT token..."):
                                success, result, token_type = generate_jwt_token(
                                    jwt_email.strip(), jwt_password.strip()
                                )
                                if success:
                                    st.success("✅ JWT token generated successfully!")

                                    # Create a container for the token display
                                    token_container = st.container()
                                    with token_container:
                                        st.write("**Your JWT Token:**")

                                        # Display token in a text area (read-only)
                                        st.text_area(
                                            "Access Token:",
                                            value=result,
                                            height=100,
                                            help="Copy this token for API calls or WebSocket connections",
                                            key="generated_jwt_token",
                                        )

                                        # Copy functionality with JavaScript
                                        col1, col2 = st.columns([1, 1])
                                        with col1:
                                            copy_button = st.button(
                                                "📋 Copy Token",
                                                key="copy_jwt_token",
                                                use_container_width=True,
                                            )
                                        with col2:
                                            copy_auth_header = st.button(
                                                "📋 Copy Auth Header",
                                                key="copy_auth_header",
                                                use_container_width=True,
                                            )

                                        if copy_button:
                                            # JavaScript copy functionality
                                            copy_js = f"""
                                            <script>
                                            function copyToClipboard() {{
                                                const text = `{result}`;
                                                navigator.clipboard.writeText(text).then(function() {{
                                                    console.log('Token copied to clipboard');
                                                }}).catch(function(err) {{
                                                    console.error('Could not copy text: ', err);
                                                    // Fallback: select the text area
                                                    const textArea = document.getElementById('generated_jwt_token');
                                                    if (textArea) {{
                                                        textArea.select();
                                                        textArea.setSelectionRange(0, 99999); // For mobile devices
                                                    }}
                                                }});
                                            }}
                                            copyToClipboard();
                                            </script>
                                            """
                                            st.components.v1.html(copy_js, height=0)
                                            st.success("✅ Token copied to clipboard!")
                                            st.info(
                                                "💡 **Fallback:** If automatic copy failed, select text in the box above and copy (Ctrl+C)"
                                            )

                                        if copy_auth_header:
                                            # JavaScript copy functionality for auth header
                                            auth_header = f"Authorization: Bearer {result}"
                                            copy_auth_js = f"""
                                            <script>
                                            function copyAuthToClipboard() {{
                                                const text = `{auth_header}`;
                                                navigator.clipboard.writeText(text).then(function() {{
                                                    console.log('Auth header copied to clipboard');
                                                }}).catch(function(err) {{
                                                    console.error('Could not copy auth header: ', err);
                                                }});
                                            }}
                                            copyAuthToClipboard();
                                            </script>
                                            """
                                            st.components.v1.html(copy_auth_js, height=0)
                                            st.success(
                                                "✅ Authorization header copied to clipboard!"
                                            )
                                            st.code(f"Authorization: Bearer {result}")
                                            st.info(
                                                "💡 **Fallback:** If automatic copy failed, select text in the code box above and copy (Ctrl+C)"
                                            )

                                        # Show usage examples
                                        st.divider()
                                        st.write("**Usage Examples:**")

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**WebSocket Connection:**")
                                            st.code(
                                                f"ws://your-server:8000/ws?token={result[:20]}..."
                                            )

                                        with col2:
                                            st.write("**API Call:**")
                                            st.code(
                                                f"""curl -H "Authorization: Bearer {result[:20]}..." \\
  {BACKEND_API_URL}/api/users"""
                                            )

                                        st.write("**Full Token (for copying):**")
                                        st.code(result)
                                else:
                                    st.error(f"❌ Failed to generate token: {result}")
                        else:
                            st.error("Please enter both email and password")

            # Registration info
            with st.expander("📝 New User Registration"):
                st.info("New users can register using the backend API:")
                st.code(f"POST {BACKEND_API_URL}/auth/register")
                st.caption("💡 Email/password registration available")

            # Show auth configuration status
            if auth_config:
                with st.expander("⚙️ Auth Configuration"):
                    st.write("**Available Methods:**")
                    st.write("• Email/Password: ✅ Enabled")
                    st.write("• Registration: ✅ Enabled")
            else:
                st.caption("⚠️ Could not load auth configuration from backend")


# ---- Health Check Functions ---- #
@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid too many requests
def get_system_health():
    """Get comprehensive system health from backend."""
    logger.info("🏥 Performing system health check")
    start_time = time.time()

    try:
        # First try the simple readiness check with shorter timeout
        logger.debug("🔍 Checking backend readiness...")
        response = requests.get(f"{BACKEND_API_URL}/readiness", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend readiness check passed")
            # Backend is responding, now try the full health check with longer timeout
            try:
                logger.debug("🔍 Performing full health check...")
                health_response = requests.get(f"{BACKEND_API_URL}/health", timeout=30)
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


# ---- Helper Functions ---- #
def get_data(endpoint: str, require_auth: bool = False):
    """Helper function to get data from the backend API with retry logic."""
    logger.debug(f"📡 GET request to endpoint: {endpoint}")
    start_time = time.time()

    # Check authentication if required
    if require_auth and not st.session_state.get("authenticated", False):
        logger.warning(f"❌ Authentication required for endpoint: {endpoint}")
        st.error(f"🔒 Authentication required to access {endpoint}")
        return None

    max_retries = 3
    base_delay = 1
    headers = get_auth_headers() if require_auth else {}

    for attempt in range(max_retries):
        try:
            logger.debug(f"📡 Attempt {attempt + 1}/{max_retries} for GET {endpoint}")
            response = requests.get(f"{BACKEND_API_URL}{endpoint}", headers=headers)

            # Handle authentication errors
            if response.status_code == 401:
                logger.error(f"❌ Authentication failed for {endpoint}")
                st.error("🔒 Authentication failed. Please login again.")
                logout()  # Clear invalid auth state
                return None
            elif response.status_code == 403:
                logger.error(f"❌ Access forbidden for {endpoint}")
                st.error("🔒 Access forbidden. You don't have permission for this resource.")
                return None

            response.raise_for_status()
            duration = time.time() - start_time
            logger.info(f"✅ GET {endpoint} successful in {duration:.3f}s")
            return response.json()
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"⚠️ GET {endpoint} attempt {attempt + 1} failed in {duration:.3f}s, retrying in {delay}s: {str(e)}"
                )
                time.sleep(delay)
                continue
            else:
                logger.error(
                    f"❌ GET {endpoint} failed after {max_retries} attempts in {duration:.3f}s: {e}"
                )
                if not require_auth:  # Only show connection error for public endpoints
                    st.error(
                        f"Could not connect to the backend at `{BACKEND_API_URL}`. Please ensure it's running. Error: {e}"
                    )
                return None


def post_data(
    endpoint: str,
    params: dict | None = None,
    json_data: dict | None = None,
    require_auth: bool = False,
):
    """Helper function to post data to the backend API."""
    logger.debug(f"📤 POST request to endpoint: {endpoint} with params: {params}")
    start_time = time.time()

    # Check authentication if required
    if require_auth and not st.session_state.get("authenticated", False):
        logger.warning(f"❌ Authentication required for endpoint: {endpoint}")
        st.error(f"🔒 Authentication required to access {endpoint}")
        return None

    headers = get_auth_headers() if require_auth else {}

    try:
        kwargs = {"headers": headers}
        if params:
            kwargs["params"] = params
        if json_data:
            kwargs["json"] = json_data

        response = requests.post(f"{BACKEND_API_URL}{endpoint}", **kwargs)

        # Handle authentication errors
        if response.status_code == 401:
            logger.error(f"❌ Authentication failed for {endpoint}")
            st.error("🔒 Authentication failed. Please login again.")
            logout()  # Clear invalid auth state
            return None
        elif response.status_code == 403:
            logger.error(f"❌ Access forbidden for {endpoint}")
            st.error("🔒 Access forbidden. You don't have permission for this resource.")
            return None

        # Handle specific HTTP status codes before raising for status
        if response.status_code == 409:
            duration = time.time() - start_time
            logger.error(f"❌ POST {endpoint} failed with 409 Conflict in {duration:.3f}s")
            # Try to get the specific error message from the response
            try:
                error_data = response.json()
                error_message = error_data.get("message", "Resource already exists")
                st.error(f"❌ {error_message}")
            except:
                st.error("❌ Resource already exists. Please check your input and try again.")
            return None

        response.raise_for_status()
        duration = time.time() - start_time
        logger.info(f"✅ POST {endpoint} successful in {duration:.3f}s")
        return response.json()
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"❌ POST {endpoint} failed in {duration:.3f}s: {e}")
        st.error(f"Error posting to backend: {e}")
        return None


def delete_data(endpoint: str, params: dict | None = None, require_auth: bool = False):
    """Helper function to delete data from the backend API."""
    logger.debug(f"🗑️ DELETE request to endpoint: {endpoint} with params: {params}")
    start_time = time.time()

    # Check authentication if required
    if require_auth and not st.session_state.get("authenticated", False):
        logger.warning(f"❌ Authentication required for endpoint: {endpoint}")
        st.error(f"🔒 Authentication required to access {endpoint}")
        return None

    headers = get_auth_headers() if require_auth else {}

    try:
        response = requests.delete(f"{BACKEND_API_URL}{endpoint}", params=params, headers=headers)

        # Handle authentication errors
        if response.status_code == 401:
            logger.error(f"❌ Authentication failed for {endpoint}")
            st.error("🔒 Authentication failed. Please login again.")
            logout()  # Clear invalid auth state
            return None
        elif response.status_code == 403:
            logger.error(f"❌ Access forbidden for {endpoint}")
            st.error("🔒 Access forbidden. You don't have permission for this resource.")
            return None

        response.raise_for_status()
        duration = time.time() - start_time
        logger.info(f"✅ DELETE {endpoint} successful in {duration:.3f}s")
        return response.json()
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"❌ DELETE {endpoint} failed in {duration:.3f}s: {e}")
        st.error(f"Error deleting from backend: {e}")
        return None


# ---- Streamlit App Configuration ---- #
logger.info("🎨 Configuring Streamlit app...")
st.set_page_config(
    page_title="Friend-Lite Dashboard", layout="wide", initial_sidebar_state="expanded"
)

# Initialize authentication state
init_auth_state()

# Check for authentication token in URL parameters
check_auth_from_url()

st.title("Friend-Lite Dashboard")
logger.info("📊 Dashboard initialized")

# Inject custom CSS for conversation box using Streamlit theme variables
st.markdown(
    """
    <style>
    .conversation-box {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid var(--secondary-background-color);
        border-radius: 5px;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        font-size: 1.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar with Authentication and Health Checks ---- #
# Show authentication first
show_auth_sidebar()

with st.sidebar:
    st.header("🔍 System Health")
    logger.debug("🔍 Loading system health sidebar...")

    with st.expander("Service Status", expanded=True):
        # Get system health from backend
        with st.spinner("Checking system health..."):
            health_data = get_system_health()

            if health_data.get("overall_healthy", False):
                st.success(f"🟢 System Status: {health_data.get('status', 'Unknown').title()}")
                logger.info("🟢 System health check passed")
            else:
                st.error(f"🔴 System Status: {health_data.get('status', 'Unknown').title()}")
                logger.warning(
                    f"🔴 System health check failed: {health_data.get('error', 'Unknown error')}"
                )

            # Show individual services with better formatting
            services = health_data.get("services", {})
            for service_name, service_info in services.items():
                status_text = service_info.get("status", "Unknown")

                # Format service names for better display
                display_name = service_name.replace("_", " ").title()
                if service_name == "speech_to_text":
                    display_name = "Speech to Text"
                elif service_name == "audioai":
                    display_name = "AudioAI"

                st.write(f"**{display_name}:** {status_text}")
                logger.debug(f"Service {service_name}: {status_text}")

                # Show provider info if available
                if "provider" in service_info:
                    st.caption(f"Provider: {service_info['provider']}")
                    logger.debug(f"Service {service_name} provider: {service_info['provider']}")

                # Show type info if available
                if "type" in service_info:
                    st.caption(f"Type: {service_info['type']}")
                    logger.debug(f"Service {service_name} type: {service_info['type']}")

                # Show additional info if available
                if "models" in service_info:
                    st.caption(f"Models available: {service_info['models']}")
                    logger.debug(f"Service {service_name} models: {service_info['models']}")
                if "uri" in service_info:
                    st.caption(f"URI: {service_info['uri']}")
                    logger.debug(f"Service {service_name} URI: {service_info['uri']}")

    if st.button("🔄 Refresh Health Check"):
        logger.info("🔄 Manual health check refresh requested")
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Close Conversation Section
    st.header("🔒 Close Conversation")
    logger.debug("🔒 Loading close conversation section...")

    with st.expander("Active Clients & Close Conversation", expanded=True):
        # Get active clients
        logger.debug("📡 Fetching active clients...")
        active_clients_data = get_data("/api/active_clients", require_auth=True)
        clients = (
            active_clients_data["clients"]
            if active_clients_data and active_clients_data.get("clients")
            else {}
        )

        if clients:
            logger.info(f"📊 Found {len(clients)} accessible clients")

            # Check if user is authenticated to show appropriate messages
            if st.session_state.get("authenticated", False):
                user_info = st.session_state.get("user_info", {})
                is_admin = (
                    user_info.get("is_superuser", False) if isinstance(user_info, dict) else False
                )

                if not is_admin and len(clients) == 0:
                    st.info("🔍 No active clients found for your account.")
                    st.caption(
                        "💡 **Tip:** Connect an audio client with your user ID to see it here."
                    )
                elif not is_admin:
                    st.caption("ℹ️ You can only see and manage your own conversations.")

            # Show active clients with conversation status
            for client_info in clients:
                client_id = client_info.get("client_id")
                logger.debug(
                    f"👤 Processing client: {client_id} - Active conversation: {client_info.get('has_active_conversation', False)}"
                )

                col1, col2 = st.columns([2, 1])

                with col1:
                    if client_info.get("has_active_conversation", False):
                        st.write(f"🟢 **{client_id}** (Active conversation)")
                        if client_info.get("current_audio_uuid"):
                            st.caption(f"UUID: {client_info['current_audio_uuid'][:8]}...")
                            logger.debug(
                                f"Client {client_id} has active conversation with UUID: {client_info['current_audio_uuid']}"
                            )
                    else:
                        st.write(f"⚪ **{client_id}** (No active conversation)")
                        logger.debug(f"Client {client_id} has no active conversation")

                with col2:
                    if client_info.get("has_active_conversation", False):
                        close_btn = st.button(
                            "🔒 Close",
                            key=f"close_{client_id}",
                            help=f"Close current conversation for {client_id}",
                            type="secondary",
                        )

                        if close_btn:
                            logger.info(f"🔒 Closing conversation for client: {client_id}")
                            result = post_data(
                                f"/api/conversations/{client_id}/close", require_auth=True
                            )
                            if result:
                                st.success(f"✅ Conversation closed for {client_id}")
                                logger.info(f"✅ Successfully closed conversation for {client_id}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to close conversation for {client_id}")
                                logger.error(f"❌ Failed to close conversation for {client_id}")
                    else:
                        st.caption("No active conversation")

            if len(clients) > 0:
                st.info(
                    f"💡 **Total accessible clients:** {active_clients_data.get('active_clients_count', 0)}"
                )
        else:
            if st.session_state.get("authenticated", False):
                st.info("🔍 No active clients found for your account.")
                st.markdown(
                    """
                **To see active clients here:**
                1. Connect an audio client using your user ID
                2. Make sure to include your authentication token in the WebSocket connection
                3. Use the format: `ws://localhost:8000/ws?user_id=YOUR_USER_ID&token=YOUR_TOKEN`
                """
                )
            else:
                st.warning("🔒 Please authenticate to view your active clients.")
            logger.info("📊 No active clients found")

    st.divider()

    # Configuration Info
    with st.expander("Configuration"):
        logger.debug("🔧 Loading configuration info...")
        health_data = get_system_health()
        config = health_data.get("config", {})

        # Display transcription service info
        transcription_service = config.get("transcription_service", "Unknown")
        websocket_enabled = config.get("websocket_stt_enabled", False)

        st.code(
            f"""
Backend API: {BACKEND_API_URL}
Backend Public: {BACKEND_PUBLIC_URL}
Active Clients: {config.get('active_clients', 'Unknown')}
MongoDB URI: {config.get('mongodb_uri', 'Unknown')[:30]}...
AudioAI URL: {config.get('llm_base_url', 'Unknown')}
Qdrant URL: {config.get('qdrant_url', 'Unknown')}
Speech to Text: {transcription_service}
STT URI: {config.get('asr_uri', 'Unknown')}
Chunk Directory: {config.get('chunk_dir', 'Unknown')}
        """
        )

        # Audio connectivity test
        st.write("**Audio Endpoint Test:**")
        try:
            import requests

            test_url = f"{BACKEND_PUBLIC_URL}/audio/"
            response = requests.head(test_url, timeout=2)
            if response.status_code in [200, 404]:  # 404 is OK for directory listing
                st.success(f"✅ Audio endpoint reachable: {test_url}")
            else:
                st.error(f"❌ Audio endpoint issue (HTTP {response.status_code}): {test_url}")
        except Exception as e:
            st.error(f"❌ Cannot reach audio endpoint: {e}")
            st.caption(f"Trying URL: {BACKEND_PUBLIC_URL}/audio/")

        # Manual override option for audio URL
        st.write("**Audio URL Override:**")
        if st.button("🔧 Fix Audio URLs"):
            # Allow user to manually set the correct public URL
            st.session_state["show_url_override"] = True

        if st.session_state.get("show_url_override", False):
            custom_url = st.text_input(
                "Custom Backend Public URL",
                value=BACKEND_PUBLIC_URL,
                help="Enter the URL that your browser can access (e.g., http://100.99.62.5:8000)",
            )
            if st.button("Apply Custom URL"):
                st.session_state["custom_backend_url"] = custom_url
                st.session_state["show_url_override"] = False
                st.success(f"✅ Audio URLs will now use: {custom_url}")
                st.rerun()

        logger.debug(f"🔧 Configuration displayed - Backend API: {BACKEND_API_URL}")

# Show warning if system is unhealthy
health_data = get_system_health()
if not health_data.get("overall_healthy", False):
    st.error("⚠️ Some critical services are unavailable. The dashboard may not function properly.")
    logger.warning("⚠️ System is unhealthy - some services unavailable")

# Show authentication status and guidance
if not st.session_state.get("authenticated", False):
    st.info(
        "🔒 **Authentication Required:** Some features require authentication. Please login using the sidebar to access user management, protected conversations, and admin functions."
    )
else:
    user_info = st.session_state.get("user_info", {})
    st.success(
        f"✅ **Authenticated as:** {user_info.get('name', 'Unknown User')} - You have access to all features."
    )

# ---- Main Content ---- #
logger.info("📋 Loading main dashboard tabs...")
# Check if user is admin to show debug tab
is_admin = False
if st.session_state.get("authenticated", False):
    user_info = st.session_state.get("user_info", {})
    if isinstance(user_info, dict):
        is_admin = user_info.get("is_superuser", False)

    # Check if the token has superuser privileges by trying an admin endpoint
    if not is_admin:
        try:
            test_response = get_data("/api/users", require_auth=True)
            if test_response and isinstance(test_response, list) and len(test_response) > 0:
                # Find the current user in the response
                current_user_email = user_info.get("email")
                for user in test_response:
                    if user.get("email") == current_user_email and user.get("is_superuser"):
                        is_admin = True
                        break
            logger.info(
                f"🔧 Admin test via /api/users: response_length={len(test_response) if test_response else 0}, is_admin={is_admin}"
            )
        except Exception as e:
            logger.warning(f"🔧 Admin test failed: {e}")

# Debug: Show admin detection status
if st.session_state.get("authenticated", False):
    user_info = st.session_state.get("user_info", {})
    st.sidebar.caption(f"🔧 Admin status: {'✅ Admin' if is_admin else '❌ Regular user'}")
    # Add debug info to help troubleshoot
    with st.sidebar.expander("🔧 Debug User Info", expanded=False):
        st.write("User Info Type:", type(user_info))
        if isinstance(user_info, dict):
            st.write("is_superuser value:", user_info.get("is_superuser", "NOT_FOUND"))
            st.write("All user_info keys:", list(user_info.keys()) if user_info else "Empty dict")
        st.write("Session authenticated:", st.session_state.get("authenticated", False))
        st.write("Final is_admin:", is_admin)

# Create tabs based on admin status
if is_admin:
    tab_convos, tab_mem, tab_users, tab_manage, tab_debug = st.tabs(
        [
            "Conversations",
            "Memories",
            "User Management",
            "Conversation Management",
            "🔧 System State",
        ]
    )
else:
    tab_convos, tab_mem, tab_users, tab_manage = st.tabs(
        ["Conversations", "Memories", "User Management", "Conversation Management"]
    )
    tab_debug = None  # Set to None for non-admin users

with tab_convos:
    logger.debug("🗨️ Loading conversations tab...")
    st.header("Latest Conversations")

    # Initialize session state for refresh tracking
    if "refresh_timestamp" not in st.session_state:
        st.session_state.refresh_timestamp = 0

    # Add debug mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Refresh Conversations"):
            logger.info("🔄 Manual conversation refresh requested")
            st.session_state.refresh_timestamp = int(time.time())
            st.session_state.refresh_random = random.randint(1000, 9999)
            st.rerun()
    with col2:
        debug_mode = st.checkbox(
            "🔧 Debug Mode",
            help="Show original audio files instead of cropped versions",
            key="debug_mode",
        )
        if debug_mode:
            logger.debug("🔧 Debug mode enabled")

    # Generate cache-busting parameter based on session state
    if st.session_state.refresh_timestamp > 0:
        random_component = getattr(st.session_state, "refresh_random", 0)
        cache_buster = f"?t={st.session_state.refresh_timestamp}&r={random_component}"
        st.info("🔄 Audio files refreshed - cache cleared for latest versions")
        logger.info("🔄 Audio cache busting applied")
    else:
        cache_buster = ""

    logger.debug("📡 Fetching conversations data...")
    conversations = get_data("/api/conversations", require_auth=True)

    if conversations:
        logger.info(
            f"📊 Loaded {len(conversations) if isinstance(conversations, list) else 'grouped'} conversations"
        )

        # Check if conversations is the new grouped format or old format
        if isinstance(conversations, dict) and "conversations" in conversations:
            # New grouped format
            logger.debug("📊 Processing conversations in new grouped format")
            conversations_data = conversations["conversations"]

            for client_id, client_conversations in conversations_data.items():
                logger.debug(
                    f"👤 Processing conversations for client: {client_id} ({len(client_conversations)} conversations)"
                )
                st.subheader(f"👤 {client_id}")

                for convo in client_conversations:
                    logger.debug(f"🗨️ Processing conversation: {convo.get('audio_uuid', 'unknown')}")

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        # Format timestamp for better readability
                        ts = datetime.fromtimestamp(convo["timestamp"])
                        st.write(f"**Timestamp:**")
                        st.write(ts.strftime("%Y-%m-%d %H:%M:%S"))

                        # Show Audio UUID
                        audio_uuid = convo.get("audio_uuid", "N/A")
                        st.write(f"**Audio UUID:**")
                        st.code(audio_uuid, language=None)

                        # Show identified speakers
                        speakers = convo.get("speakers_identified", [])
                        if speakers:
                            st.write(f"**Speakers:**")
                            for speaker in speakers:
                                st.write(f"🎤 `{speaker}`")
                            logger.debug(f"🎤 Speakers identified: {speakers}")

                        # Show audio duration info if available
                        cropped_duration = convo.get("cropped_duration")
                        if cropped_duration:
                            st.write(f"**Cropped Duration:**")
                            st.write(f"⏱️ {cropped_duration:.1f}s")

                            # Show speech segments count
                            speech_segments = convo.get("speech_segments", [])
                            if speech_segments:
                                st.write(f"**Speech Segments:**")
                                st.write(f"🗣️ {len(speech_segments)} segments")
                                logger.debug(f"🗣️ Speech segments: {len(speech_segments)}")

                    with col2:
                        # Display conversation transcript with new format
                        transcript = convo.get("transcript", [])
                        if transcript:
                            logger.debug(
                                f"📝 Displaying transcript with {len(transcript)} segments"
                            )
                            st.write("**Conversation:**")
                            conversation_text = ""
                            for segment in transcript:
                                speaker = segment.get("speaker", "Unknown")
                                text = segment.get("text", "")
                                start_time = segment.get("start", 0.0)
                                end_time = segment.get("end", 0.0)

                                # Format timing if available
                                timing_info = ""
                                if start_time > 0 or end_time > 0:
                                    timing_info = f" [{start_time:.1f}s - {end_time:.1f}s]"

                                conversation_text += (
                                    f"<b>{speaker}</b>{timing_info}: {text}<br><br>"
                                )

                            # Display in a scrollable container with max height
                            st.markdown(
                                f'<div class="conversation-box">{conversation_text}</div>',
                                unsafe_allow_html=True,
                            )

                        # Smart audio display logic
                        audio_path = convo.get("audio_path")
                        cropped_audio_path = convo.get("cropped_audio_path")

                        if audio_path:
                            # Determine which audio to show
                            if debug_mode:
                                # Debug mode: always show original
                                selected_audio_path = audio_path
                                audio_label = "🔧 **Original Audio** (Debug Mode)"
                                logger.debug(f"🔧 Debug mode: showing original audio: {audio_path}")
                            elif cropped_audio_path:
                                # Normal mode: prefer cropped if available
                                selected_audio_path = cropped_audio_path
                                audio_label = "🎵 **Cropped Audio** (Silence Removed)"
                                logger.debug(
                                    f"🎵 Normal mode: showing cropped audio: {cropped_audio_path}"
                                )
                            else:
                                # Fallback: show original if no cropped version
                                selected_audio_path = audio_path
                                audio_label = "🎵 **Original Audio** (No cropped version available)"
                                logger.debug(
                                    f"🎵 Fallback: showing original audio (no cropped version): {audio_path}"
                                )

                            # Display audio with label and cache-busting
                            st.write(audio_label)
                            # Use custom URL if set, otherwise use detected URL
                            backend_url = st.session_state.get(
                                "custom_backend_url", BACKEND_PUBLIC_URL
                            )
                            audio_url = f"{backend_url}/audio/{selected_audio_path}{cache_buster}"

                            # Test audio accessibility
                            try:
                                import requests

                                test_response = requests.head(audio_url, timeout=2)
                                if test_response.status_code == 200:
                                    st.audio(audio_url, format="audio/wav")
                                    logger.debug(f"🎵 Audio URL accessible: {audio_url}")
                                else:
                                    st.error(
                                        f"❌ Audio file not accessible (HTTP {test_response.status_code})"
                                    )
                                    st.code(f"URL: {audio_url}")
                                    logger.error(
                                        f"🎵 Audio URL not accessible: {audio_url} (HTTP {test_response.status_code})"
                                    )
                            except Exception as e:
                                st.error(f"❌ Cannot reach audio file: {str(e)}")
                                st.code(f"URL: {audio_url}")
                                logger.error(f"🎵 Audio URL error: {audio_url} - {e}")

                            # Show additional info in debug mode or when both versions exist
                            if debug_mode and cropped_audio_path:
                                st.caption(f"💡 Cropped version available: {cropped_audio_path}")
                            elif not debug_mode and cropped_audio_path:
                                st.caption(f"💡 Enable debug mode to hear original with silence")

                        # Display memory information if available
                        memories = convo.get("memories", [])
                        if memories:
                            st.write("**🧠 Memories Created:**")
                            memory_count = len(memories)
                            st.write(
                                f"📊 {memory_count} memory{'ies' if memory_count != 1 else ''} extracted from this conversation"
                            )

                            # Show memory details in an expandable section
                            with st.expander(
                                f"📋 View Memory Details ({memory_count} items)", expanded=False
                            ):
                                # Fetch actual memory content from the API with higher limit (cached)
                                cache_key = f"memories_{st.session_state.get('user', {}).get('id', 'unknown')}"
                                if cache_key not in st.session_state or st.button(
                                    "🔄 Refresh Memories",
                                    key=f"refresh_{cache_key}_{hash(str(memories))}",
                                ):
                                    st.session_state[cache_key] = get_data(
                                        "/api/memories?limit=500", require_auth=True
                                    )
                                user_memories_response = st.session_state.get(cache_key)
                                memory_contents = {}

                                if user_memories_response and "memories" in user_memories_response:
                                    for mem in user_memories_response["memories"]:
                                        memory_contents[mem.get("id")] = mem.get(
                                            "memory", "No content available"
                                        )

                                for i, memory in enumerate(memories):
                                    memory_id = memory.get("memory_id", "Unknown")
                                    status = memory.get("status", "unknown")
                                    created_at = memory.get("created_at", "Unknown")

                                    # Get actual memory content
                                    memory_text = memory_contents.get(
                                        memory_id, "Memory content not found"
                                    )

                                    # Display each memory with content
                                    st.write(f"**Memory {i+1}:**")

                                    # Show memory content in a highlighted box
                                    if (
                                        memory_text
                                        and memory_text != "Memory content not found"
                                        and memory_text != "No content available"
                                    ):
                                        st.info(f"💭 {memory_text}")
                                    else:
                                        st.warning(f"🔍 ID: `{memory_id}`")
                                        st.caption(
                                            "Memory content not available - this may be a transcript-based fallback"
                                        )

                                    st.caption(f"📅 Created: {created_at}")

                                    # Show status badge
                                    if status == "created":
                                        st.success(f"✅ {status}")
                                    else:
                                        st.info(f"ℹ️ {status}")

                                    if i < len(memories) - 1:  # Add separator between memories
                                        st.markdown("---")
                        else:
                            # Show when no memories are available
                            if convo.get("has_memory") is False:
                                st.caption("🔍 No memories extracted from this conversation yet")

                    st.divider()
        else:
            # Old format - single list of conversations
            logger.debug("📊 Processing conversations in old format")
            for convo in conversations:
                logger.debug(f"🗨️ Processing conversation: {convo.get('audio_uuid', 'unknown')}")

                col1, col2 = st.columns([1, 4])
                with col1:
                    # Format timestamp for better readability
                    ts = datetime.fromtimestamp(convo["timestamp"])
                    st.write(f"**Timestamp:**")
                    st.write(ts.strftime("%Y-%m-%d %H:%M:%S"))

                    # Show client_id with better formatting
                    client_id = convo.get("client_id", "N/A")
                    if client_id.startswith("client_"):
                        st.write(f"**Client ID:**")
                        st.write(f"`{client_id}`")
                    else:
                        st.write(f"**User ID:**")
                        st.write(f"👤 `{client_id}`")

                    # Show Audio UUID
                    audio_uuid = convo.get("audio_uuid", "N/A")
                    st.write(f"**Audio UUID:**")
                    st.code(audio_uuid, language=None)

                    # Show identified speakers
                    speakers = convo.get("speakers_identified", [])
                    if speakers:
                        st.write(f"**Speakers:**")
                        for speaker in speakers:
                            st.write(f"🎤 `{speaker}`")

                with col2:
                    # Display conversation transcript with new format
                    transcript = convo.get("transcript", [])
                    if transcript:
                        logger.debug(f"📝 Displaying transcript with {len(transcript)} segments")
                        st.write("**Conversation:**")
                        conversation_text = ""
                        for segment in transcript:
                            speaker = segment.get("speaker", "Unknown")
                            text = segment.get("text", "")
                            start_time = segment.get("start", 0.0)
                            end_time = segment.get("end", 0.0)

                            # Format timing if available
                            timing_info = ""
                            if start_time > 0 or end_time > 0:
                                timing_info = f" [{start_time:.1f}s - {end_time:.1f}s]"

                            conversation_text += f"<b>{speaker}</b>{timing_info}: {text}<br><br>"

                        # Display in a scrollable container with max height
                        st.markdown(
                            f'<div class="conversation-box">{conversation_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        # Fallback for old format
                        old_transcript = convo.get("transcription", "No transcript available.")
                        st.text_area(
                            "Transcription",
                            old_transcript,
                            height=150,
                            disabled=True,
                            key=f"transcript_{convo['_id']}",
                        )

                    # Smart audio display logic (same as above)
                    audio_path = convo.get("audio_path")
                    cropped_audio_path = convo.get("cropped_audio_path")

                    if audio_path:
                        # Determine which audio to show
                        if debug_mode:
                            # Debug mode: always show original
                            selected_audio_path = audio_path
                            audio_label = "🔧 **Original Audio** (Debug Mode)"
                            logger.debug(f"🔧 Debug mode: showing original audio: {audio_path}")
                        elif cropped_audio_path:
                            # Normal mode: prefer cropped if available
                            selected_audio_path = cropped_audio_path
                            audio_label = "🎵 **Cropped Audio** (Silence Removed)"
                            logger.debug(
                                f"🎵 Normal mode: showing cropped audio: {cropped_audio_path}"
                            )
                        else:
                            # Fallback: show original if no cropped version
                            selected_audio_path = audio_path
                            audio_label = "🎵 **Original Audio** (No cropped version available)"
                            logger.debug(
                                f"🎵 Fallback: showing original audio (no cropped version): {audio_path}"
                            )

                        # Display audio with label and cache-busting
                        st.write(audio_label)
                        # Use custom URL if set, otherwise use detected URL
                        backend_url = st.session_state.get("custom_backend_url", BACKEND_PUBLIC_URL)
                        audio_url = f"{backend_url}/audio/{selected_audio_path}{cache_buster}"

                        # Test audio accessibility
                        try:
                            import requests

                            test_response = requests.head(audio_url, timeout=2)
                            if test_response.status_code == 200:
                                st.audio(audio_url, format="audio/wav")
                                logger.debug(f"🎵 Audio URL accessible: {audio_url}")
                            else:
                                st.error(
                                    f"❌ Audio file not accessible (HTTP {test_response.status_code})"
                                )
                                st.code(f"URL: {audio_url}")
                                logger.error(
                                    f"🎵 Audio URL not accessible: {audio_url} (HTTP {test_response.status_code})"
                                )
                        except Exception as e:
                            st.error(f"❌ Cannot reach audio file: {str(e)}")
                            st.code(f"URL: {audio_url}")
                            logger.error(f"🎵 Audio URL error: {audio_url} - {e}")

                        # Show additional info in debug mode or when both versions exist
                        if debug_mode and cropped_audio_path:
                            st.caption(f"💡 Cropped version available: {cropped_audio_path}")
                        elif not debug_mode and cropped_audio_path:
                            st.caption(f"💡 Enable debug mode to hear original with silence")

                    # Display memory information if available (same as grouped format)
                    memories = convo.get("memories", [])
                    if memories:
                        st.write("**🧠 Memories Created:**")
                        memory_count = len(memories)
                        st.write(
                            f"📊 {memory_count} memory{'ies' if memory_count != 1 else ''} extracted from this conversation"
                        )

                        # Show memory details in an expandable section
                        with st.expander(
                            f"📋 View Memory Details ({memory_count} items)", expanded=False
                        ):
                            # Fetch actual memory content from the API
                            user_memories_response = get_data("/api/memories", require_auth=True)
                            memory_contents = {}

                            if user_memories_response and "memories" in user_memories_response:
                                for mem in user_memories_response["memories"]:
                                    memory_contents[mem.get("id")] = mem.get(
                                        "memory", "No content available"
                                    )

                            for i, memory in enumerate(memories):
                                memory_id = memory.get("memory_id", "Unknown")
                                status = memory.get("status", "unknown")
                                created_at = memory.get("created_at", "Unknown")

                                # Get actual memory content
                                memory_text = memory_contents.get(
                                    memory_id, "Memory content not found"
                                )

                                # Display each memory with content
                                st.write(f"**Memory {i+1}:**")

                                # Show memory content in a highlighted box
                                if (
                                    memory_text
                                    and memory_text != "Memory content not found"
                                    and memory_text != "No content available"
                                ):
                                    st.info(f"💭 {memory_text}")
                                else:
                                    st.warning(f"🔍 ID: `{memory_id}`")
                                    st.caption(
                                        "Memory content not available - this may be a transcript-based fallback"
                                    )

                                st.caption(f"📅 Created: {created_at}")

                                # Show status badge
                                if status == "created":
                                    st.success(f"✅ {status}")
                                else:
                                    st.info(f"ℹ️ {status}")

                                if i < len(memories) - 1:  # Add separator between memories
                                    st.markdown("---")
                    else:
                        # Show when no memories are available
                        if convo.get("has_memory") is False:
                            st.caption("🔍 No memories extracted from this conversation yet")

                st.divider()
    elif conversations is not None:
        st.info("No conversations found. The backend is connected but the database might be empty.")
        logger.info("📊 No conversations found in database")

with tab_mem:
    logger.debug("🧠 Loading memories tab...")
    st.header("Memories")

    # Use session state for selected user if available
    default_user = st.session_state.get("selected_user", "")

    # User selection for memories
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input(
            "Enter username to view memories:",
            value=default_user,
            placeholder="e.g., john_doe, alice123",
        )
    with col2:
        st.write("")  # Spacer
        refresh_mem_btn = st.button("Load Data", key="refresh_memories")

    # Clear the session state after using it
    if "selected_user" in st.session_state:
        del st.session_state["selected_user"]

    if refresh_mem_btn:
        logger.info("🔄 Manual memories refresh requested")
        st.rerun()

    # Get memories based on user selection
    if user_id_input.strip():
        logger.info(f"🧠 Loading data for user: {user_id_input.strip()}")
        st.info(f"Showing data for user: **{user_id_input.strip()}**")

        # Load memories
        with st.spinner("Loading memories..."):
            logger.debug(f"📡 Fetching memories for user: {user_id_input.strip()}")
            memories_response = get_data(
                f"/api/memories?user_id={user_id_input.strip()}", require_auth=True
            )

        # Handle the API response format with "results" wrapper for memories
        if (
            memories_response
            and isinstance(memories_response, dict)
            and "results" in memories_response
        ):
            memories = memories_response["results"]
            logger.debug(
                f"🧠 Memories response has 'results' wrapper, extracted {len(memories)} memories"
            )
        else:
            memories = memories_response
            logger.debug(f"🧠 Memories response format: {type(memories_response)}")

    else:
        # Show instruction to enter a username
        memories = None
        logger.debug("👆 No user ID provided, showing instructions")
        st.info("👆 Please enter a username above to view their memories.")
        st.markdown("💡 **Tip:** You can find existing usernames in the 'User Management' tab.")

    # Admin Debug Section - Show before regular memories
    if st.session_state.get("authenticated", False):
        user_info = st.session_state.get("user_info", {})

        # Check if user is admin (look for is_superuser in different possible locations)
        is_admin = False
        if isinstance(user_info, dict):
            is_admin = user_info.get("is_superuser", False)

        # Alternative: Check if the token has superuser privileges by trying an admin endpoint
        if not is_admin:
            try:
                test_response = get_data("/api/users", require_auth=True)
                is_admin = test_response is not None
            except:
                pass

        if is_admin:
            st.subheader("🔧 Admin Debug: All Memories")
            logger.debug("🔧 Admin user detected, showing admin debug section")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🧠 View All User Memories (Admin)", key="admin_all_memories"):
                    logger.info("📋 Admin: Loading all memories for all users")
                    st.session_state["show_admin_memories"] = True

            with col2:
                if st.session_state.get("show_admin_memories", False):
                    if st.button("❌ Hide Admin View", key="hide_admin_views"):
                        st.session_state["show_admin_memories"] = False
                        st.rerun()

            # Show admin memories view if requested
            if st.session_state.get("show_admin_memories", False):
                with st.spinner("Loading memories..."):
                    logger.debug("📋 Fetching memories for admin view")

                    # Use the working user memories endpoint since admin is a user too
                    user_memories_response = get_data("/api/memories?limit=500", require_auth=True)

                    if user_memories_response and "memories" in user_memories_response:
                        # Get current user info
                        user_info = st.session_state.get("user", {})
                        user_id = user_info.get("id", "unknown")
                        user_email = user_info.get("email", "unknown")

                        memories = user_memories_response["memories"]

                        # Format as admin response for compatibility with existing UI
                        admin_memories_response = {
                            "memories": [
                                {
                                    "id": memory.get("id"),
                                    "memory": memory.get("memory", "No content"),
                                    "user_id": user_id,
                                    "owner_email": user_email,
                                    "created_at": memory.get("created_at"),
                                    "client_id": memory.get("metadata", {}).get(
                                        "client_id", "unknown"
                                    ),
                                    "metadata": memory.get("metadata", {}),
                                }
                                for memory in memories
                            ],
                            "user_memories": (
                                {
                                    user_id: [
                                        {
                                            "memory": memory.get("memory", "No content"),
                                            "created_at": memory.get("created_at"),
                                            "client_id": memory.get("metadata", {}).get(
                                                "client_id", "unknown"
                                            ),
                                            "owner_email": user_email,
                                        }
                                        for memory in memories
                                    ]
                                }
                                if memories
                                else {}
                            ),
                            "total_memories": len(memories),
                            "total_users": 1 if memories else 0,
                            "stats": {
                                "users_with_memories": [user_id] if memories else [],
                                "client_ids_with_memories": [],
                            },
                        }
                    else:
                        admin_memories_response = None

                if admin_memories_response:
                    logger.info(
                        f"📋 Admin memories: Loaded {admin_memories_response.get('total_memories', 0)} memories from {admin_memories_response.get('total_users', 0)} users"
                    )

                    # Display summary stats including debug info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Users", admin_memories_response.get("total_users", 0))
                    with col2:
                        st.metric(
                            "Total Memories", admin_memories_response.get("total_memories", 0)
                        )
                    with col3:
                        stats = admin_memories_response.get("stats", {})
                        st.metric(
                            "Debug Tracker",
                            "✅" if stats.get("debug_tracker_initialized") else "❌",
                        )

                    st.divider()

                    # Add view toggle
                    view_mode = st.radio(
                        "View Mode:", ["📋 By User", "🔍 All Memories"], horizontal=True
                    )

                    if view_mode == "📋 By User":
                        # Display memories grouped by user
                        user_memories = admin_memories_response.get("user_memories", {})
                        stats = admin_memories_response.get("stats", {})

                        if user_memories:
                            st.write("### 👥 Memories by User")

                            # Show debug info
                            users_with_memories = stats.get("users_with_memories", [])
                            client_ids_with_memories = stats.get("client_ids_with_memories", [])

                            if users_with_memories:
                                st.caption(
                                    f"Found users: {', '.join(users_with_memories[:5])}{'...' if len(users_with_memories) > 5 else ''}"
                                )

                            for user_id, user_memory_list in user_memories.items():
                                memory_count = len(user_memory_list)

                                # Get user info from first memory if available
                                user_email = "Unknown"
                                if user_memory_list:
                                    user_email = user_memory_list[0].get("owner_email", user_id)

                                # User header with collapsible section
                                with st.expander(
                                    f"👤 {user_email} ({user_id}) - {memory_count} memories",
                                    expanded=False,
                                ):
                                    if user_memory_list:
                                        # Show first 10 memories for this user
                                        memories_to_show = user_memory_list[:10]

                                        for i, memory in enumerate(memories_to_show):
                                            memory_text = memory.get("memory", "No content")
                                            created_at = memory.get("created_at", "Unknown")
                                            client_id = memory.get("client_id", "Unknown")

                                            st.write(
                                                f"**{i+1}.** {memory_text[:200]}{'...' if len(memory_text) > 200 else ''}"
                                            )
                                            st.caption(f"📅 {created_at} | 🔌 {client_id}")

                                            if i < len(memories_to_show) - 1:
                                                st.markdown("---")

                                        if memory_count > 10:
                                            st.info(f"... and {memory_count - 10} more memories")
                                    else:
                                        st.info("No memories found for this user.")

                            if client_ids_with_memories:
                                st.write("### 🔌 Debug: Client IDs Found")
                                st.caption(
                                    f"Client IDs: {', '.join(client_ids_with_memories[:10])}{'...' if len(client_ids_with_memories) > 10 else ''}"
                                )

                        else:
                            st.info("No memories found across all users.")

                    else:
                        # Display all memories in flat view
                        memories = admin_memories_response.get("memories", [])

                        if memories:
                            st.write("### 🧠 All User Memories")

                            # Create a searchable/filterable view
                            search_term = st.text_input(
                                "🔍 Search memories", placeholder="Enter text to search..."
                            )

                            if search_term:
                                filtered_memories = [
                                    m
                                    for m in memories
                                    if search_term.lower() in m.get("memory", "").lower()
                                    or search_term.lower() in m.get("owner_email", "").lower()
                                    or search_term.lower() in m.get("user_id", "").lower()
                                ]
                                st.caption(
                                    f"Showing {len(filtered_memories)} memories matching '{search_term}'"
                                )
                            else:
                                filtered_memories = memories
                                st.caption(f"Showing all {len(memories)} memories")

                            # Display memories in a nice format
                            for i, memory in enumerate(
                                filtered_memories[:50]
                            ):  # Limit to 50 for performance
                                with st.container():
                                    # Memory header
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.write(f"**Memory {i+1}**")
                                    with col2:
                                        st.caption(
                                            f"👤 {memory.get('owner_email', memory.get('user_id', 'Unknown'))}"
                                        )
                                    with col3:
                                        st.caption(f"📅 {memory.get('created_at', 'Unknown')}")

                                    # Memory content
                                    memory_text = memory.get("memory", "No content")
                                    st.write(memory_text)

                                    # Memory metadata
                                    with st.expander("🔍 Memory Details", expanded=False):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(
                                                f"**User ID:** {memory.get('user_id', 'Unknown')}"
                                            )
                                            st.write(
                                                f"**Owner Email:** {memory.get('owner_email', 'Unknown')}"
                                            )
                                            st.write(
                                                f"**Client ID:** {memory.get('client_id', 'Unknown')}"
                                            )
                                        with col2:
                                            st.write(
                                                f"**Audio UUID:** {memory.get('audio_uuid', 'Unknown')}"
                                            )
                                            st.write(
                                                f"**Memory ID:** {memory.get('id', memory.get('memory_id', 'Unknown'))}"
                                            )
                                            metadata = memory.get("metadata", {})
                                            if metadata:
                                                st.write(
                                                    f"**Source:** {metadata.get('source', 'Unknown')}"
                                                )

                                    st.divider()

                            if len(filtered_memories) > 50:
                                st.info(
                                    f"Showing first 50 memories. Total: {len(filtered_memories)}"
                                )

                        else:
                            st.info("No memories found across all users.")

                else:
                    logger.error("❌ Failed to load admin memories")
                    st.error("❌ Failed to load admin memories. You may not have admin privileges.")

            st.divider()

    # Display Memories Section
    if memories is not None:
        logger.debug("🧠 Displaying memories section...")
        st.subheader("🧠 Discovered Memories")

        if memories:
            logger.info(f"🧠 Displaying {len(memories)} memories for user {user_id_input.strip()}")

            # Add view options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"Found **{len(memories)}** memories for user **{user_id_input.strip()}**"
                )
            with col2:
                view_mode = st.selectbox(
                    "View Mode:", ["Standard View", "Transcript Analysis"], key="memory_view_mode"
                )

            if view_mode == "Standard View":
                # Original view
                df = pd.DataFrame(memories)

                # Make the dataframe more readable
                if "created_at" in df.columns:
                    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                # Reorder and rename columns for clarity - handle both "memory" and "text" fields
                display_cols = {"id": "Memory ID", "created_at": "Created At"}

                # Check which memory field exists and add it to display columns
                if "memory" in df.columns:
                    display_cols["memory"] = "Memory"
                    logger.debug("🧠 Using 'memory' field for display")
                elif "text" in df.columns:
                    display_cols["text"] = "Memory"
                    logger.debug("🧠 Using 'text' field for display")

                # Filter for columns that exist in the dataframe
                cols_to_display = [col for col in display_cols.keys() if col in df.columns]

                if cols_to_display:
                    logger.debug(f"🧠 Displaying columns: {cols_to_display}")
                    st.dataframe(
                        df[cols_to_display].rename(columns=display_cols),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    logger.error(
                        f"⚠️ Unexpected memory data format - missing expected fields. Available columns: {list(df.columns)}"
                    )
                    st.error("⚠️ Unexpected memory data format - missing expected fields")
                    st.write("Debug info - Available columns:", list(df.columns))

            else:  # Transcript Analysis View
                with st.spinner("Loading memories with transcript analysis..."):
                    enriched_response = get_data(
                        f"/api/memories/with-transcripts?user_id={user_id_input.strip()}",
                        require_auth=True,
                    )

                if enriched_response:
                    enriched_memories = enriched_response.get("memories", [])

                    if enriched_memories:
                        # Create enhanced dataframe for transcript analysis
                        analysis_data = []
                        for memory in enriched_memories:
                            analysis_data.append(
                                {
                                    "Audio UUID": (
                                        memory.get("audio_uuid", "N/A")[:12] + "..."
                                        if memory.get("audio_uuid")
                                        else "N/A"
                                    ),
                                    "Memory Text": (
                                        memory.get("memory_text", "")[:100] + "..."
                                        if len(memory.get("memory_text", "")) > 100
                                        else memory.get("memory_text", "")
                                    ),
                                    "Transcript": (
                                        memory.get("transcript", "")[:100] + "..."
                                        if memory.get("transcript")
                                        and len(memory.get("transcript", "")) > 100
                                        else (
                                            memory.get("transcript", "N/A")[:100]
                                            if memory.get("transcript")
                                            else "N/A"
                                        )
                                    ),
                                    "Transcript Chars": memory.get("transcript_length", 0),
                                    "Memory Chars": memory.get("memory_length", 0),
                                    "Compression %": f"{memory.get('compression_ratio', 0)}%",
                                    "Client ID": memory.get("client_id", "N/A"),
                                    "Created": (
                                        memory.get("created_at", "N/A")[:19]
                                        if memory.get("created_at")
                                        else "N/A"
                                    ),
                                }
                            )

                        # Display the enhanced table
                        if analysis_data:
                            analysis_df = pd.DataFrame(analysis_data)
                            st.dataframe(analysis_df, use_container_width=True, hide_index=True)

                            # Show detailed expandable views
                            st.subheader("📋 Detailed Memory Analysis")

                            for i, memory in enumerate(enriched_memories):
                                audio_uuid = memory.get("audio_uuid", "unknown")
                                memory_text = memory.get("memory_text", "")
                                transcript = memory.get("transcript", "")
                                compression_ratio = memory.get("compression_ratio", 0)

                                # Create meaningful title
                                title_text = (
                                    memory_text[:50] + "..."
                                    if len(memory_text) > 50
                                    else memory_text
                                )
                                if not title_text.strip():
                                    title_text = f"Memory {i+1}"

                                with st.expander(
                                    f"🧠 {title_text} | {compression_ratio}% compression",
                                    expanded=False,
                                ):
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("**🎤 Original Transcript**")
                                        if transcript and transcript.strip():
                                            st.text_area(
                                                f"Transcript ({len(transcript)} chars):",
                                                value=transcript,
                                                height=200,
                                                disabled=True,
                                                key=f"transcript_{i}",
                                            )
                                        else:
                                            st.info("No transcript available")

                                    with col2:
                                        st.markdown("**🧠 Extracted Memory**")
                                        if memory_text and memory_text.strip():
                                            st.text_area(
                                                f"Memory ({len(memory_text)} chars):",
                                                value=memory_text,
                                                height=200,
                                                disabled=True,
                                                key=f"memory_text_{i}",
                                            )
                                        else:
                                            st.warning("No memory text")

                                    # Additional details
                                    st.markdown("**📊 Metadata**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Audio UUID",
                                            (
                                                audio_uuid[:12] + "..."
                                                if audio_uuid and len(audio_uuid) > 12
                                                else audio_uuid or "N/A"
                                            ),
                                        )
                                    with col2:
                                        st.metric("Client ID", memory.get("client_id", "N/A"))
                                    with col3:
                                        st.metric("User Email", memory.get("user_email", "N/A"))
                        else:
                            st.info("No enriched memory data available")
                    else:
                        st.info("No memories with transcript data found")
                else:
                    st.error("Failed to load enriched memory data")
        else:
            logger.info(f"🧠 No memories found for user {user_id_input.strip()}")
            st.info("No memories found for this user.")


with tab_users:
    st.header("User Management")

    # Create User Section
    st.subheader("Create New User")
    with st.form("create_user_form"):
        st.write("Create a new user with an email and a temporary password.")
        new_user_email = st.text_input("New User Email:", placeholder="e.g., john.doe@example.com")
        new_user_password = st.text_input("Temporary Password:", type="password", value="changeme")
        create_user_submitted = st.form_submit_button("Create User")

        if create_user_submitted:
            if new_user_email.strip() and new_user_password.strip():
                create_data = {
                    "email": new_user_email.strip(),
                    "password": new_user_password.strip(),
                }
                # This endpoint requires authentication
                result = post_data("/api/create_user", json_data=create_data, require_auth=True)
                if result:
                    st.success(f"✅ User '{new_user_email.strip()}' created successfully!")
                    st.rerun()
                # Note: Error handling for 409 Conflict (user exists) is now handled in post_data function
            else:
                st.error("❌ Please provide both email and password.")

    st.divider()

    # List Users Section
    st.subheader("Existing Users")
    col1, col2 = st.columns([1, 1])
    with col1:
        refresh_users_btn = st.button("Refresh Users", key="refresh_users")

    if refresh_users_btn:
        st.rerun()

    users = get_data("/api/users", require_auth=True)

    if users:
        st.write(f"**Total Users:** {len(users)}")

        # Debug: Show first user structure (temporary)
        with st.expander("🐛 Debug: User Data Structure", expanded=False):
            if users:
                st.write("**First user data structure:**")
                st.json(users[0])
                st.caption("💡 This shows the actual fields returned by the API")

        # Initialize session state for delete confirmation
        if "delete_confirmation" not in st.session_state:
            st.session_state.delete_confirmation = {}

        # Display users in a nice format
        for index, user in enumerate(users):
            # The API returns 'id' (ObjectId), 'email', 'display_name', etc.
            # Use display_name if available, otherwise email, otherwise the ID
            user_display = user.get("display_name") or user.get("email", "Unknown User")
            user_db_id = str(user.get("id", "unknown"))  # MongoDB ObjectId as string
            # Create unique key using both user_db_id and index to avoid duplicates
            unique_key = f"{user_db_id}_{index}"

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👤 **{user_display}**")
                st.caption(f"Email: {user.get('email', 'No email')}")
                st.caption(f"ID: {user_db_id}")

            with col2:
                # Check if we're in confirmation mode for this user (use db_id as key)
                if user_db_id in st.session_state.delete_confirmation:
                    # Show confirmation dialog in a container
                    with st.container():
                        st.error("⚠️ **Confirm Deletion**")
                        st.write(f"Delete user **{user_display}** and optionally:")

                        # Checkboxes for what to delete
                        delete_conversations = st.checkbox(
                            "🗨️ Delete all conversations",
                            key=f"conv_{unique_key}",
                            help="Permanently delete all audio recordings and transcripts",
                        )
                        delete_memories = st.checkbox(
                            "🧠 Delete all memories",
                            key=f"mem_{unique_key}",
                            help="Permanently delete all extracted memories from conversations",
                        )

                        # Action buttons
                        col_cancel, col_confirm = st.columns([1, 1])

                        with col_cancel:
                            if st.button(
                                "❌ Cancel",
                                key=f"cancel_{unique_key}",
                                use_container_width=True,
                                type="secondary",
                            ):
                                del st.session_state.delete_confirmation[user_db_id]
                                st.rerun()

                        with col_confirm:
                            if st.button(
                                "🗑️ Confirm Delete",
                                key=f"confirm_{unique_key}",
                                use_container_width=True,
                                type="primary",
                            ):
                                # Build delete parameters - use MongoDB ObjectId
                                params = {
                                    "user_id": user_db_id,  # MongoDB ObjectId as string
                                    "delete_conversations": delete_conversations,
                                    "delete_memories": delete_memories,
                                }

                                # This endpoint requires authentication
                                result = delete_data("/api/delete_user", params, require_auth=True)
                                if result:
                                    deleted_data = result.get("deleted_data", {})
                                    message = result.get(
                                        "message", f"User '{user_display}' deleted"
                                    )
                                    st.success(message)

                                    # Show detailed deletion info
                                    if (
                                        deleted_data.get("conversations_deleted", 0) > 0
                                        or deleted_data.get("memories_deleted", 0) > 0
                                    ):
                                        st.info(
                                            f"📊 Deleted: {deleted_data.get('conversations_deleted', 0)} conversations, {deleted_data.get('memories_deleted', 0)} memories"
                                        )

                                    del st.session_state.delete_confirmation[user_db_id]
                                    st.rerun()

                        if delete_conversations or delete_memories:
                            st.caption(
                                "⚠️ Selected data will be **permanently deleted** and cannot be recovered!"
                            )
                else:
                    # Show normal delete button
                    delete_btn = st.button("🗑️ Delete", key=f"delete_{unique_key}", type="secondary")
                    if delete_btn:
                        st.session_state.delete_confirmation[user_db_id] = True
                        st.rerun()

            st.divider()

    elif users is not None:
        st.info("No users found in the system.")

    st.divider()

    # Quick Actions Section
    st.subheader("Quick Actions")
    st.write("**View User Memories:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        quick_user_id = st.text_input(
            "User ID to view memories:", placeholder="Enter user ID", key="quick_view_user"
        )
    with col2:
        st.write("")  # Spacer
        view_memories_btn = st.button("View Memories", key="view_memories")

    if view_memories_btn and quick_user_id.strip():
        # Switch to memories tab with this user
        st.session_state["selected_user"] = quick_user_id.strip()
        st.info(f"Switch to the 'Memories' tab to view memories for user: {quick_user_id.strip()}")

    # Tips section
    st.subheader("💡 Tips")
    st.markdown(
        """
    - **User IDs** should be unique identifiers (e.g., usernames, email prefixes)
    - Users are automatically created when they connect with audio if they don't exist
    - **Delete Options:**
      - **User Account**: Always deleted when you click delete
      - **🗨️ Conversations**: Check to delete all audio recordings and transcripts
      - **🧠 Memories**: Check to delete all extracted memories from conversations
      - Mix and match: You can delete just conversations, just memories, or both
    - Use the 'Memories' tab to view specific user memories
    """
    )

    # Authentication information
    st.subheader("🔐 Authentication System")
    if st.session_state.get("authenticated", False):
        st.success("✅ You are authenticated and can use all user management features.")
        user_info = st.session_state.get("user_info", {})
        st.info(f"**Current User:** {user_info.get('name', 'Unknown')}")
        st.info(f"**Auth Method:** {st.session_state.get('auth_method', 'unknown').title()}")
    else:
        st.warning("🔒 Authentication required for user management operations.")
        st.markdown(
            """
        **How to authenticate:**
        1. **Email/Password**: Use the login form in the sidebar if you have an account
        2. **Manual Token**: If you have a JWT token, paste it in the manual entry section

        **Note:** The backend requires authentication for:
        - Creating new users
        - Deleting users and their data
        - WebSocket audio connections
        """
        )

        st.markdown("**Authentication Configuration:**")
        st.code(
            f"""
# Required environment variables for backend:
AUTH_SECRET_KEY=your-secret-key
        """
        )

        st.caption("💡 Email/password authentication is enabled by default")

with tab_manage:
    st.header("Conversation Management")

    st.subheader("🔒 Close Current Conversation")

    # Check if user is authenticated and show appropriate message
    if st.session_state.get("authenticated", False):
        user_info = st.session_state.get("user_info", {})
        is_admin = user_info.get("is_superuser", False) if isinstance(user_info, dict) else False

        if is_admin:
            st.write("Close the current active conversation for any connected client.")
        else:
            st.write("Close the current active conversation for your connected clients.")

        # Get active clients for the dropdown
        active_clients_data = get_data("/api/active_clients", require_auth=True)

        if active_clients_data and active_clients_data.get("clients"):
            clients = active_clients_data["clients"]

            # Filter to only clients with active conversations
            active_conversations = {
                client_info.get("client_id"): client_info
                for client_info in clients
                if client_info.get("has_active_conversation", False)
            }

            if active_conversations:
                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_client = st.selectbox(
                        "Select client to close conversation:",
                        options=list(active_conversations.keys()),
                        format_func=lambda x: f"{x} (UUID: {active_conversations[x].get('current_audio_uuid', 'N/A')[:8]}...)",
                    )

                with col2:
                    st.write("")  # Spacer
                    close_conversation_btn = st.button(
                        "🔒 Close Conversation", key="close_conv_main", type="primary"
                    )

                if close_conversation_btn and selected_client:
                    result = post_data(
                        f"/api/conversations/{selected_client}/close", require_auth=True
                    )
                    if result:
                        st.success(
                            f"✅ Successfully closed conversation for client '{selected_client}'!"
                        )
                        st.info(f"📋 {result.get('message', 'Conversation closed')}")
                        time.sleep(1)  # Brief pause before refresh
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to close conversation for client '{selected_client}'")
            else:
                if len(clients) > 0:
                    st.info("🔍 No clients with active conversations found.")
                    st.caption(
                        "💡 Your connected clients don't have active conversations at the moment."
                    )
                else:
                    st.info("🔍 No connected clients found for your account.")
                    st.caption(
                        "💡 Connect an audio client with your user ID to manage conversations."
                    )

            # Show all clients status (only if there are clients)
            if len(clients) > 0:
                with st.expander("All Connected Clients Status"):
                    for client_info in clients:
                        client_id = client_info.get("client_id")
                        status_icon = (
                            "🟢" if client_info.get("has_active_conversation", False) else "⚪"
                        )
                        st.write(
                            f"{status_icon} **{client_id}** - {'Active conversation' if client_info.get('has_active_conversation', False) else 'No active conversation'}"
                        )
                        if client_info.get("current_audio_uuid"):
                            st.caption(f"   Audio UUID: {client_info['current_audio_uuid']}")

                    # Show ownership info for non-admin users
                    if not is_admin:
                        st.caption(
                            "ℹ️ You can only see and manage clients that belong to your account."
                        )
        else:
            st.info("🔍 No accessible clients found for your account.")
            st.markdown(
                """
            **To connect an audio client:**
            1. Use your user ID when connecting: `user_id=YOUR_USER_ID`
            2. Include your authentication token in the WebSocket connection
            3. Example: `ws://localhost:8000/ws?user_id=YOUR_USER_ID&token=YOUR_TOKEN`
            """
            )

            if st.session_state.get("auth_token"):
                st.info(
                    "💡 Your authentication token is available - see the WebSocket connection info below."
                )
            else:
                st.warning(
                    "⚠️ Please authenticate first to get your token for audio client connections."
                )
    else:
        st.warning("🔒 Authentication required to manage conversations.")
        st.markdown(
            """
        **Please authenticate using the sidebar to:**
        - View your active audio clients
        - Close conversations for your clients
        - Manage your conversation data
        """
        )
        st.info("👆 Use the authentication options in the sidebar to get started.")

    st.divider()

    st.subheader("Add Speaker to Conversation")
    st.write("Add speakers to conversations even if they haven't spoken yet.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        audio_uuid_input = st.text_input("Audio UUID:", placeholder="Enter the audio UUID")
    with col2:
        speaker_id_input = st.text_input("Speaker ID:", placeholder="e.g., speaker_1, john_doe")
    with col3:
        st.write("")  # Spacer
        add_speaker_btn = st.button("Add Speaker", key="add_speaker")

    if add_speaker_btn:
        if audio_uuid_input.strip() and speaker_id_input.strip():
            result = post_data(
                f"/api/conversations/{audio_uuid_input.strip()}/speakers",
                params={"speaker_id": speaker_id_input.strip()},
                require_auth=True,
            )
            if result:
                st.success(f"Speaker '{speaker_id_input.strip()}' added to conversation!")
        else:
            st.error("Please enter both Audio UUID and Speaker ID")

    st.divider()

    st.subheader("Update Transcript Segment")
    st.write("Modify speaker identification or timing information for transcript segments.")

    col1, col2 = st.columns([1, 1])
    with col1:
        update_audio_uuid = st.text_input(
            "Audio UUID:", placeholder="Enter the audio UUID", key="update_uuid"
        )
        segment_index = st.number_input("Segment Index:", min_value=0, value=0, step=1)
        new_speaker = st.text_input(
            "New Speaker ID (optional):", placeholder="Leave empty to keep current"
        )

    with col2:
        start_time = st.number_input(
            "Start Time (seconds):", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
        end_time = st.number_input(
            "End Time (seconds):", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
        update_segment_btn = st.button("Update Segment", key="update_segment")

    if update_segment_btn:
        if update_audio_uuid.strip():
            params = {}
            if new_speaker.strip():
                params["speaker_id"] = new_speaker.strip()
            if start_time > 0:
                params["start_time"] = start_time
            if end_time > 0:
                params["end_time"] = end_time

            if params:
                # Use requests.put for this endpoint
                try:
                    response = requests.put(
                        f"{BACKEND_API_URL}/api/conversations/{update_audio_uuid.strip()}/transcript/{segment_index}",
                        params=params,
                        headers=get_auth_headers(),
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success("Transcript segment updated successfully!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error updating segment: {e}")
            else:
                st.warning("Please specify at least one field to update")
        else:
            st.error("Please enter the Audio UUID")

    st.divider()

    st.subheader("💡 Schema Information")
    st.markdown(
        """
    **New Conversation Schema:**
    ```json
    {
        "audio_uuid": "unique_identifier",
        "audio_path": "path/to/audio/file.wav",
        "client_id": "user_or_client_id",
        "timestamp": 1234567890,
        "transcript": [
            {
                "speaker": "speaker_1",
                "text": "Hello, how are you?",
                "start": 0.0,
                "end": 3.2
            },
            {
                "speaker": "speaker_2",
                "text": "I'm good, thanks!",
                "start": 3.3,
                "end": 5.0
            }
        ],
        "speakers_identified": ["speaker_1", "speaker_2"]
    }
    ```
    """
    )

    st.info(
        "💡 **Tip**: You can find Audio UUIDs in the conversation details on the 'Conversations' tab."
    )

    st.divider()

    # Authentication info for WebSocket connections
    st.subheader("🔐 Authentication & WebSocket Connections")
    if st.session_state.get("authenticated", False):
        auth_token = st.session_state.get("auth_token", "")
        st.success(
            "✅ You are authenticated. Audio clients can use your token for WebSocket connections."
        )

        with st.expander("WebSocket Connection Info"):
            st.markdown("**For audio clients, use one of these WebSocket URLs:**")
            st.code(
                f"""
# Opus audio stream (with authentication):
ws://localhost:8000/ws?token={auth_token[:20]}...

# PCM audio stream (with authentication):
ws://localhost:8000/ws_pcm?token={auth_token[:20]}...

# Or include in Authorization header:
Authorization: Bearer {auth_token[:20]}...
            """
            )
            st.caption("⚠️ Keep your token secure and don't share it publicly!")

        st.info("🎵 **Audio clients must now authenticate** to connect to WebSocket endpoints.")
    else:
        st.warning("🔒 WebSocket audio connections now require authentication.")
        st.markdown(
            """
        **Important Changes:**
        - All WebSocket endpoints (`/ws` and `/ws_pcm`) now require authentication
        - Audio clients must include a JWT token in the connection
        - Tokens can be passed via query parameter (`?token=...`) or Authorization header
        - Get a token by logging in via the sidebar or using the backend auth endpoints
        """
        )

        st.info(
            "👆 **Log in using the sidebar** to get your authentication token for audio clients."
        )

# System State Tab
if tab_debug is not None:
    with tab_debug:
        st.header("🔧 System State & Failure Recovery")
        st.caption("Real-time system monitoring and debug information")

        # Check authentication like other tabs
        if not st.session_state.get("authenticated", False):
            st.warning("🔒 Please log in to access system monitoring features")
        else:
            # Show immediate system status
            st.info("💡 **Click the buttons below to load different system monitoring sections**")

            # Get debug system tracker data
            try:
                tracker = get_debug_tracker()
                dashboard_data = tracker.get_dashboard_data()
                system_metrics = dashboard_data["system_metrics"]
                recent_transactions = dashboard_data["recent_transactions"]
                recent_issues = dashboard_data["recent_issues"]

                # Quick system status check (always visible)
                with st.container():
                    st.subheader("⚡ System Overview")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Transactions", system_metrics["total_transactions"])

                    with col2:
                        st.metric("Active", system_metrics["active_transactions"])

                    with col3:
                        st.metric("Failed", system_metrics["failed_transactions"])

                    with col4:
                        st.metric("Completed", system_metrics["completed_transactions"])

                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Active WebSockets", system_metrics["active_websockets"])
                with col2:
                    st.metric("Audio Chunks", system_metrics["total_audio_chunks"])
                with col3:
                    st.metric("Transcriptions", system_metrics["total_transcriptions"])
                with col4:
                    st.metric("Memories Created", system_metrics["total_memories"])

                # System uptime and activity
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Uptime (hours)", f"{system_metrics['uptime_hours']:.1f}")
                with col2:
                    st.metric("Active Users", dashboard_data["active_users"])
                with col3:
                    st.metric("Stalled", system_metrics["stalled_transactions"])

                st.divider()

                # Recent issues (pipeline problems)
                if recent_issues:
                    st.subheader("🚨 Recent Issues")
                    st.warning(f"Found {len(recent_issues)} recent issues that need attention:")

                    issues_data = []
                    for issue in recent_issues:
                        issues_data.append(
                            {
                                "Timestamp": issue["timestamp"][:19].replace("T", " "),
                                "Transaction": issue["transaction_id"][:8],
                                "User": (
                                    issue["user_id"][-6:]
                                    if len(issue["user_id"]) > 6
                                    else issue["user_id"]
                                ),
                                "Issue": issue["issue"],
                            }
                        )

                    issues_df = pd.DataFrame(issues_data)
                    st.dataframe(issues_df, use_container_width=True)
                else:
                    st.success("✅ No recent issues detected!")

                st.divider()

                # Recent transactions
                st.subheader("📋 Recent Transactions")

                if recent_transactions:
                    transaction_data = []
                    for t in recent_transactions:
                        status_emoji = {
                            "in_progress": "🔄",
                            "completed": "✅",
                            "failed": "❌",
                            "stalled": "⏰",
                        }.get(t["status"], "❓")

                        transaction_data.append(
                            {
                                "Status": f"{status_emoji} {t['status'].title()}",
                                "Stage": t["current_stage"].replace("_", " ").title(),
                                "User": t["user_id"],
                                "Created": t["created_at"][:19].replace("T", " "),
                                "Issue": t["issue"] or "",
                            }
                        )

                    df = pd.DataFrame(transaction_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent transactions")

            except Exception as e:
                st.error(f"❌ Error loading system data: {e}")
                st.write(
                    "Debug tracker may not be initialized yet or there may be a configuration issue."
                )

            # Refresh button
            if st.button("🔄 Refresh System Stats"):
                st.rerun()

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📈 Load Debug Stats", key="load_debug_stats"):
                    st.session_state["debug_stats_loaded"] = True
            with col2:
                if st.button("🔄 Refresh Debug Data", key="refresh_debug_data"):
                    # Clear cached data to force refresh
                    if "debug_stats_loaded" in st.session_state:
                        del st.session_state["debug_stats_loaded"]
                    if "debug_sessions_loaded" in st.session_state:
                        del st.session_state["debug_sessions_loaded"]
                    st.rerun()

            if st.session_state.get("debug_stats_loaded", False):
                with st.spinner("Loading debug statistics..."):
                    try:
                        debug_stats = get_data("/api/debug/memory/stats", require_auth=True)

                        if debug_stats:
                            stats = debug_stats.get("stats", {})

                            st.success("✅ Memory processing statistics loaded successfully")

                            # Display key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Sessions", stats.get("total_sessions", 0))
                            with col2:
                                success_rate = stats.get("success_rate", 0) or 0
                                st.metric(
                                    "Success Rate",
                                    f"{success_rate:.1f}%",
                                    delta=f"{'✅' if success_rate > 80 else '⚠️' if success_rate > 50 else '❌'}",
                                )
                            with col3:
                                avg_time = stats.get("avg_processing_time_seconds", 0) or 0
                                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
                            with col4:
                                failed = stats.get("failed_sessions", 0) or 0
                                st.metric(
                                    "Failed Sessions",
                                    failed,
                                    delta=f"{'✅' if failed == 0 else '⚠️'}",
                                )

                            # Show additional metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Memories", stats.get("total_memories", 0) or 0)
                            with col2:
                                st.metric(
                                    "Successful Sessions", stats.get("successful_sessions", 0) or 0
                                )
                            with col3:
                                memories_per_session = stats.get("memories_per_session", 0) or 0
                                st.metric("Memories per Session", f"{memories_per_session:.1f}")

                            # Show detailed stats
                            with st.expander("📋 Detailed Statistics", expanded=False):
                                st.json(stats)
                        else:
                            st.error("❌ Failed to load debug statistics - No data received")
                            st.caption(
                                "This could indicate an authentication issue or the debug endpoint is not available"
                            )
                    except Exception as e:
                        st.error(f"❌ Error loading debug statistics: {str(e)}")
                        st.caption("Check the backend logs for more details")

            st.divider()

            # Recent Sessions Section
            st.subheader("📝 Recent Memory Sessions")

            col1, col2 = st.columns([1, 1])
            with col1:
                session_limit = st.number_input(
                    "Number of sessions to load:", min_value=5, max_value=100, value=20, step=5
                )
            with col2:
                if st.button("📋 Load Recent Sessions", key="load_debug_sessions"):
                    st.session_state["debug_sessions_loaded"] = True

            if st.session_state.get("debug_sessions_loaded", False):
                with st.spinner("Loading recent memory sessions..."):
                    debug_sessions = get_data(
                        f"/api/debug/memory/sessions?limit={session_limit}", require_auth=True
                    )

                    if debug_sessions:
                        sessions = debug_sessions.get("sessions", [])

                        if sessions:
                            st.success(f"✅ Loaded {len(sessions)} memory sessions")

                            # Display sessions in a table
                            session_data = []
                            for session in sessions:
                                session_data.append(
                                    {
                                        "Audio UUID": session.get("audio_uuid", "N/A")[:12] + "...",
                                        "User ID": session.get("user_id", "N/A")[:12] + "...",
                                        "Status": session.get("status", "unknown"),
                                        "Processing Time": f"{session.get('processing_time', 0):.2f}s",
                                        "Transcript Length": session.get("transcript_length", 0),
                                        "Memory Count": session.get("memory_count", 0),
                                        "Created": (
                                            session.get("created_at", "N/A")[:19]
                                            if session.get("created_at")
                                            else "N/A"
                                        ),
                                    }
                                )

                            df = pd.DataFrame(session_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No memory sessions found")
                    else:
                        st.error("Failed to load memory sessions")

            st.divider()

            # Transcript vs Memory Comparison Section
            st.subheader("🔍 Transcript vs Memory Analysis")

            st.markdown(
                "Compare original transcripts with extracted memories to understand memory extraction quality."
            )

            # Add section for viewing all transcripts vs memories
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                audio_uuid_input = st.text_input(
                    "Enter Audio UUID for analysis:",
                    placeholder="e.g., 84a6fced90aa4232ac00db6bbfcf626b",
                    help="Enter the full audio UUID to analyze transcript vs memory extraction",
                )
            with col2:
                if st.button(
                    "🔍 Analyze Session",
                    key="analyze_transcript_memory",
                    disabled=not audio_uuid_input.strip(),
                ):
                    st.session_state["transcript_analysis_uuid"] = audio_uuid_input.strip()
                    st.session_state["transcript_analysis_loaded"] = True
            with col3:
                if st.button(
                    "📋 Show All Transcripts",
                    key="btn_show_all_transcripts",
                    help="Show all transcripts vs memories for comprehensive analysis",
                ):
                    st.session_state["show_all_transcripts_view"] = True
                    st.session_state["transcript_analysis_loaded"] = (
                        False  # Clear single session analysis
                    )

            if st.session_state.get("transcript_analysis_loaded", False) and st.session_state.get(
                "transcript_analysis_uuid"
            ):
                analysis_uuid = st.session_state["transcript_analysis_uuid"]

                with st.spinner(
                    f"Loading transcript vs memory analysis for {analysis_uuid[:12]}..."
                ):
                    transcript_analysis = get_data(
                        f"/api/debug/memory/transcript-vs-memory/{analysis_uuid}", require_auth=True
                    )

                    if transcript_analysis:
                        st.success(f"✅ Analysis loaded for session {analysis_uuid[:12]}...")

                        # Session Info
                        with st.expander("📋 Session Information", expanded=True):
                            session_info = transcript_analysis.get("session_info", {})
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("User", transcript_analysis.get("user_email", "N/A"))
                                st.metric("Client ID", transcript_analysis.get("client_id", "N/A"))

                            with col2:
                                success = session_info.get("memory_processing_success", False)
                                st.metric(
                                    "Processing Status", "✅ Success" if success else "❌ Failed"
                                )

                                if session_info.get("memory_processing_error"):
                                    st.error(f"Error: {session_info['memory_processing_error']}")

                            with col3:
                                analysis = transcript_analysis.get("analysis", {})
                                compression_ratio = transcript_analysis.get("memories", {}).get(
                                    "compression_ratio_percent", 0
                                )
                                st.metric("Compression Ratio", f"{compression_ratio}%")

                        # Transcript vs Memory Comparison
                        st.subheader("📝 Transcript vs Memory Comparison")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🎤 Original Transcript")
                            transcript_data = transcript_analysis.get("transcript", {})

                            # Transcript statistics
                            st.markdown(
                                f"""
                            **Statistics:**
                            - **Characters:** {transcript_data.get('character_count', 0):,}
                            - **Words:** {transcript_data.get('word_count', 0):,}
                            - **Segments:** {transcript_data.get('segment_count', 0)}
                            """
                            )

                            # Full conversation text
                            full_conversation = transcript_data.get("full_conversation", "")
                            if full_conversation.strip():
                                st.text_area(
                                    "Full Conversation:",
                                    value=full_conversation,
                                    height=300,
                                    disabled=True,
                                    key="original_transcript",
                                )
                            else:
                                st.warning("No transcript available")

                        with col2:
                            st.markdown("### 🧠 Extracted Memories")
                            memories_data = transcript_analysis.get("memories", {})

                            # Memory statistics
                            st.markdown(
                                f"""
                            **Statistics:**
                            - **Extractions:** {memories_data.get('extraction_count', 0)}
                            - **Characters:** {memories_data.get('total_memory_characters', 0):,}
                            - **Compression:** {memories_data.get('compression_ratio_percent', 0)}%
                            """
                            )

                            # Memory extractions
                            extractions = memories_data.get("extractions", [])
                            if extractions:
                                for i, memory in enumerate(extractions):
                                    with st.expander(
                                        f"Memory {i+1}: {memory.get('memory_type', 'general')}",
                                        expanded=i == 0,
                                    ):
                                        st.markdown(
                                            f"**ID:** `{memory.get('memory_id', 'unknown')}`"
                                        )
                                        st.markdown(
                                            f"**Type:** {memory.get('memory_type', 'general')}"
                                        )

                                        memory_text = memory.get("memory_text", "")
                                        if memory_text:
                                            st.text_area(
                                                "Memory Text:",
                                                value=memory_text,
                                                height=100,
                                                disabled=True,
                                                key=f"memory_{i}",
                                            )

                                        # Show extraction prompt and LLM response in details
                                        with st.expander("🔧 Extraction Details"):
                                            if memory.get("extraction_prompt"):
                                                st.markdown("**Prompt Used:**")
                                                st.code(
                                                    memory["extraction_prompt"], language="text"
                                                )

                                            if memory.get("llm_response"):
                                                st.markdown("**Raw LLM Response:**")
                                                st.code(memory["llm_response"], language="text")
                            else:
                                analysis = transcript_analysis.get("analysis", {})
                                if analysis.get("empty_results"):
                                    st.info(
                                        "🤔 LLM determined no memorable content in this conversation"
                                    )
                                else:
                                    st.warning("No memory extractions found")

                        # Analysis Summary
                        st.subheader("📊 Analysis Summary")
                        analysis = transcript_analysis.get("analysis", {})

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            has_transcript = analysis.get("has_transcript", False)
                            st.metric("Has Transcript", "✅ Yes" if has_transcript else "❌ No")

                        with col2:
                            has_memories = analysis.get("has_memories", False)
                            st.metric("Has Memories", "✅ Yes" if has_memories else "❌ No")

                        with col3:
                            processing_successful = analysis.get("processing_successful", False)
                            st.metric(
                                "Processing OK", "✅ Yes" if processing_successful else "❌ No"
                            )

                        with col4:
                            empty_results = analysis.get("empty_results", False)
                            st.metric("Empty Results", "⚠️ Yes" if empty_results else "✅ No")

                        # Quality Assessment
                        if has_transcript and processing_successful:
                            if has_memories:
                                compression_ratio = memories_data.get(
                                    "compression_ratio_percent", 0
                                )
                                if compression_ratio > 50:
                                    st.warning(
                                        "⚠️ High compression ratio - may indicate poor memory extraction"
                                    )
                                elif compression_ratio < 5:
                                    st.warning(
                                        "⚠️ Very low compression ratio - memories may be too brief"
                                    )
                                else:
                                    st.success("✅ Good compression ratio for memory extraction")
                            elif empty_results:
                                st.info("ℹ️ LLM correctly identified no memorable content")
                            else:
                                st.error(
                                    "❌ Processing succeeded but no memories or errors recorded"
                                )

                    else:
                        st.error(f"Failed to load analysis for {analysis_uuid}")

            # Show All Transcripts vs Memories section
            if st.session_state.get("show_all_transcripts_view", False):
                st.subheader("📋 All Transcripts vs Memories Analysis")

                # Options for filtering
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    user_filter = st.text_input(
                        "Filter by User (optional):",
                        placeholder="e.g., user@example.com",
                        help="Leave empty to show all users (admin only)",
                    )
                with col2:
                    limit = st.number_input(
                        "Limit results:",
                        min_value=10,
                        max_value=500,
                        value=50,
                        step=10,
                        help="Maximum number of memories to display",
                    )
                with col3:
                    st.write("")  # Spacer
                    if st.button("🔄 Refresh Data", key="refresh_all_transcripts"):
                        st.session_state["all_transcripts_data"] = None  # Clear cache
                        st.rerun()

                # Load all transcripts vs memories
                if "all_transcripts_data" not in st.session_state:
                    with st.spinner("Loading all transcripts vs memories..."):
                        try:
                            # Use appropriate endpoint based on user permissions and filters
                            if user_filter.strip():
                                # Filter by specific user
                                endpoint = f"/api/memories/with-transcripts?user_id={user_filter.strip()}&limit={limit}"
                            else:
                                # Show all users (admin only) or current user
                                endpoint = f"/api/memories/with-transcripts?limit={limit}"

                            all_data = get_data(endpoint, require_auth=True)
                            st.session_state["all_transcripts_data"] = all_data

                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")
                            st.session_state["all_transcripts_data"] = None

                # Display the data
                if st.session_state.get("all_transcripts_data"):
                    data = st.session_state["all_transcripts_data"]
                    memories = data.get("memories", [])

                    if memories:
                        st.success(f"✅ Loaded {len(memories)} memories with transcript analysis")

                        # Summary statistics
                        total_memories = len(memories)
                        memories_with_transcripts = sum(
                            1
                            for m in memories
                            if m.get("transcript") and m.get("transcript").strip()
                        )
                        memories_without_transcripts = total_memories - memories_with_transcripts
                        avg_compression = (
                            sum(m.get("compression_ratio", 0) for m in memories) / total_memories
                            if total_memories > 0
                            else 0
                        )

                        # Display summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Memories", total_memories)
                        with col2:
                            st.metric("With Transcripts", memories_with_transcripts)
                        with col3:
                            st.metric("Without Transcripts", memories_without_transcripts)
                        with col4:
                            st.metric("Avg Compression", f"{avg_compression:.1f}%")

                        # Filter and search options
                        st.subheader("🔍 Filter Options")
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            search_term = st.text_input(
                                "Search in memories/transcripts:",
                                placeholder="Enter text to search...",
                                key="search_all_transcripts",
                            )
                        with col2:
                            show_only_with_transcripts = st.checkbox(
                                "Only with transcripts", value=False, key="filter_with_transcripts"
                            )
                        with col3:
                            compression_filter = st.selectbox(
                                "Compression filter:",
                                [
                                    "All",
                                    "High (>50%)",
                                    "Medium (10-50%)",
                                    "Low (<10%)",
                                    "Zero (0%)",
                                ],
                                key="compression_filter",
                            )

                        # Apply filters
                        filtered_memories = memories

                        if search_term:
                            filtered_memories = [
                                m
                                for m in filtered_memories
                                if (
                                    search_term.lower() in m.get("memory_text", "").lower()
                                    or search_term.lower() in m.get("transcript", "").lower()
                                )
                            ]

                        if show_only_with_transcripts:
                            filtered_memories = [
                                m
                                for m in filtered_memories
                                if m.get("transcript") and m.get("transcript").strip()
                            ]

                        if compression_filter != "All":
                            if compression_filter == "High (>50%)":
                                filtered_memories = [
                                    m
                                    for m in filtered_memories
                                    if m.get("compression_ratio", 0) > 50
                                ]
                            elif compression_filter == "Medium (10-50%)":
                                filtered_memories = [
                                    m
                                    for m in filtered_memories
                                    if 10 <= m.get("compression_ratio", 0) <= 50
                                ]
                            elif compression_filter == "Low (<10%)":
                                filtered_memories = [
                                    m
                                    for m in filtered_memories
                                    if 0 < m.get("compression_ratio", 0) < 10
                                ]
                            elif compression_filter == "Zero (0%)":
                                filtered_memories = [
                                    m
                                    for m in filtered_memories
                                    if m.get("compression_ratio", 0) == 0
                                ]

                        if search_term or show_only_with_transcripts or compression_filter != "All":
                            st.caption(
                                f"Showing {len(filtered_memories)} of {total_memories} memories"
                            )

                        # Display results in a table format
                        if filtered_memories:
                            st.subheader("📊 Transcript vs Memory Analysis Table")

                            # Create summary table
                            table_data = []
                            for memory in filtered_memories:
                                table_data.append(
                                    {
                                        "Audio UUID": (
                                            memory.get("audio_uuid", "N/A")[:12] + "..."
                                            if memory.get("audio_uuid")
                                            else "N/A"
                                        ),
                                        "Memory": (
                                            memory.get("memory_text", "")[:60] + "..."
                                            if len(memory.get("memory_text", "")) > 60
                                            else memory.get("memory_text", "")
                                        ),
                                        "Transcript": (
                                            memory.get("transcript", "N/A")[:60] + "..."
                                            if memory.get("transcript")
                                            and len(memory.get("transcript", "")) > 60
                                            else (
                                                memory.get("transcript", "N/A")[:60]
                                                if memory.get("transcript")
                                                else "N/A"
                                            )
                                        ),
                                        "T-Chars": memory.get("transcript_length", 0),
                                        "M-Chars": memory.get("memory_length", 0),
                                        "Compression": f"{memory.get('compression_ratio', 0):.1f}%",
                                        "Client": (
                                            memory.get("client_id", "N/A")[:8] + "..."
                                            if memory.get("client_id")
                                            else "N/A"
                                        ),
                                        "Created": (
                                            memory.get("created_at", "N/A")[:16]
                                            if memory.get("created_at")
                                            else "N/A"
                                        ),
                                    }
                                )

                            # Display table
                            df = pd.DataFrame(table_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)

                            # Detailed expandable views
                            st.subheader("🔍 Detailed Analysis")

                            for i, memory in enumerate(filtered_memories):
                                audio_uuid = memory.get("audio_uuid", "unknown")
                                memory_text = memory.get("memory_text", "")
                                transcript = memory.get("transcript", "")
                                compression_ratio = memory.get("compression_ratio", 0)
                                client_id = memory.get("client_id", "unknown")

                                # Create meaningful title
                                title_text = (
                                    memory_text[:50] + "..."
                                    if len(memory_text) > 50
                                    else memory_text
                                )
                                if not title_text.strip():
                                    title_text = f"Memory {i+1}"

                                # Color code based on compression ratio
                                if compression_ratio > 50:
                                    status_emoji = "🔴"  # High compression
                                elif compression_ratio > 10:
                                    status_emoji = "🟡"  # Medium compression
                                elif compression_ratio > 0:
                                    status_emoji = "🟢"  # Good compression
                                else:
                                    status_emoji = "⚪"  # No compression

                                with st.expander(
                                    f"{status_emoji} {title_text} | {compression_ratio:.1f}% | {client_id[:8]}...",
                                    expanded=False,
                                ):
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("**🎤 Original Transcript**")
                                        if transcript and transcript.strip():
                                            st.text_area(
                                                f"Transcript ({len(transcript)} chars):",
                                                value=transcript,
                                                height=200,
                                                disabled=True,
                                                key=f"all_transcript_{i}",
                                            )
                                        else:
                                            st.info("No transcript available")

                                    with col2:
                                        st.markdown("**🧠 Extracted Memory**")
                                        if memory_text and memory_text.strip():
                                            st.text_area(
                                                f"Memory ({len(memory_text)} chars):",
                                                value=memory_text,
                                                height=200,
                                                disabled=True,
                                                key=f"all_memory_{i}",
                                            )
                                        else:
                                            st.warning("No memory text")

                                    # Additional metadata
                                    st.markdown("**📋 Metadata**")
                                    metadata_col1, metadata_col2 = st.columns(2)
                                    with metadata_col1:
                                        st.write(f"**Audio UUID:** `{audio_uuid}`")
                                        st.write(f"**Client ID:** `{client_id}`")
                                    with metadata_col2:
                                        st.write(f"**Created:** {memory.get('created_at', 'N/A')}")
                                        st.write(f"**User:** {memory.get('user_email', 'N/A')}")
                        else:
                            st.info("No memories match the current filters")
                    else:
                        st.warning("No memories found with the current filters")
                else:
                    st.error("Failed to load transcript vs memory data")

                # Add option to close the view
                if st.button("❌ Close Analysis", key="close_all_transcripts"):
                    st.session_state["show_all_transcripts_view"] = False
                    st.rerun()

            st.divider()

            # Help Section
            st.subheader("📚 Debug API Reference")

            with st.expander("🔗 Available Debug Endpoints", expanded=False):
                st.markdown(
                    """
                **Memory Debug APIs:**
                - `GET /api/debug/memory/stats` - Memory processing statistics
                - `GET /api/debug/memory/sessions` - Recent memory sessions
                - `GET /api/debug/memory/session/{uuid}` - Session details
                - `GET /api/debug/memory/transcript-vs-memory/{uuid}` - Transcript vs memory comparison
                - `GET /api/debug/memory/config` - Memory configuration
                - `GET /api/debug/memory/pipeline/{uuid}` - Processing pipeline trace

                All endpoints require authentication.
                """
                )

            st.info(
                "💡 **Tip**: Use these debug tools to monitor system performance and troubleshoot issues with memory extraction."
            )
