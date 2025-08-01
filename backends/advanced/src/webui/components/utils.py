"""
Utility functions for the Streamlit UI
"""
import logging
import time
import requests
import streamlit as st

logger = logging.getLogger("streamlit-ui")


def get_auth_headers():
    """Get authentication headers for API requests."""
    if st.session_state.get("auth_token"):
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}


def get_data(endpoint: str, require_auth: bool = False, backend_url: str = None):
    """Helper function to get data from the backend API with retry logic."""
    from .auth import logout  # Import here to avoid circular imports
    
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
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
            response = requests.get(f"{backend_api_url}{endpoint}", headers=headers)

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
                        f"Could not connect to the backend at `{backend_api_url}`. Please ensure it's running. Error: {e}"
                    )
                return None


def post_data(
    endpoint: str,
    params: dict | None = None,
    json_data: dict | None = None,
    require_auth: bool = False,
    backend_url: str = None,
    files: dict = None,
):
    """Helper function to post data to the backend API."""
    from .auth import logout  # Import here to avoid circular imports
    
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
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
        if files:
            kwargs["files"] = files

        response = requests.post(f"{backend_api_url}{endpoint}", **kwargs)

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


def delete_data(endpoint: str, params: dict | None = None, require_auth: bool = False, backend_url: str = None):
    """Helper function to delete data from the backend API."""
    from .auth import logout  # Import here to avoid circular imports
    
    backend_api_url = backend_url or st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
    
    logger.debug(f"🗑️ DELETE request to endpoint: {endpoint} with params: {params}")
    start_time = time.time()

    # Check authentication if required
    if require_auth and not st.session_state.get("authenticated", False):
        logger.warning(f"❌ Authentication required for endpoint: {endpoint}")
        st.error(f"🔒 Authentication required to access {endpoint}")
        return None

    headers = get_auth_headers() if require_auth else {}

    try:
        response = requests.delete(f"{backend_api_url}{endpoint}", params=params, headers=headers)

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