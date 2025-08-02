"""
User Management tab component for the Streamlit UI
"""
import logging

import streamlit as st

from ..utils import get_data, post_data, delete_data

logger = logging.getLogger("streamlit-ui")

def show_user_management_tab():
    """Display the user management tab with full functionality"""
    logger.debug("👥 Loading users tab...")
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
        2. **External Auth**: Some deployments support external authentication flows
        """
        )