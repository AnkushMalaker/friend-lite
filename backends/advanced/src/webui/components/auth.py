"""
Authentication functions for the Streamlit UI
"""
import json
import logging
import base64
import requests
import streamlit as st

logger = logging.getLogger("streamlit-ui")


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
def get_auth_config(backend_url: str = None):
    """Get authentication configuration from backend."""
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
    try:
        response = requests.get(f"{backend_api_url}/api/auth/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get auth config: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error getting auth config: {e}")
        return None


def check_auth_from_url(backend_url: str = None):
    """Check for authentication token in URL parameters."""
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
    try:
        # Check URL parameters for token
        query_params = st.query_params
        if "token" in query_params:
            token = query_params["token"]
            logger.info("🔐 Authentication token found in URL parameters")

            # Validate token by calling a protected endpoint
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{backend_api_url}/api/users", headers=headers, timeout=5)

            if response.status_code == 200:
                st.session_state.authenticated = True
                st.session_state.auth_token = token
                st.session_state.auth_method = "token"

                # Try to get user info from token (decode JWT payload)
                try:
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


def login_with_credentials(email, password, backend_url: str = None):
    """Login with email and password."""
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
    try:
        logger.info(f"🔐 Attempting login for email: {email}")
        response = requests.post(
            f"{backend_api_url}/auth/jwt/login",
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


def generate_jwt_token(email, password, backend_url: str = None):
    """Generate JWT token for given credentials."""
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
    try:
        logger.info(f"🔑 Generating JWT token for: {email}")
        response = requests.post(
            f"{backend_api_url}/auth/jwt/login",
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


def show_auth_sidebar(backend_url: str = None):
    """Show authentication status and controls in sidebar."""
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
    with st.sidebar:
        st.header("🔐 Authentication")

        # Get auth configuration from backend
        auth_config = get_auth_config(backend_url)

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
                                f"{backend_api_url}/api/users", headers=headers, timeout=5
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
                                    email.strip(), password.strip(), backend_url
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
                                    jwt_email.strip(), jwt_password.strip(), backend_url
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
  {backend_api_url}/api/users"""
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
                st.code(f"POST {backend_api_url}/auth/register")
                st.caption("💡 Email/password registration available")

            # Show auth configuration status
            if auth_config:
                with st.expander("⚙️ Auth Configuration"):
                    st.write("**Available Methods:**")
                    st.write("• Email/Password: ✅ Enabled")
                    st.write("• Registration: ✅ Enabled")
            else:
                st.caption("⚠️ Could not load auth configuration from backend")