*** Settings ***
Documentation    User account management and lifecycle keywords
...
...              This file contains keywords for user account creation, deletion, and management.
...              Keywords in this file should handle user-related operations, user account lifecycle,
...              and user permission management.
...
...              Examples of keywords that belong here:
...              - User account creation and deletion
...              - User management operations
...              - User permission validation
...              - User account lifecycle operations
...
...              Keywords that should NOT be in this file:
...              - Verification/assertion keywords (belong in tests)
...              - API session management (belong in session_resources.robot)
...              - Docker service management (belong in setup_resources.robot)
Library          RequestsLibrary
Library          Collections
Library          String
Variables        ../test_env.py
Resource         session_resources.robot

*** Keywords ***

Create Test User
      [Documentation]    Create a test user for testing (uses admin session)
      [Arguments]    ${session}=${EMPTY}    ${email}=${TEST_USER_EMAIL}    ${password}=${TEST_USER_PASSWORD}    ${is_superuser}=False

      # Create user
      IF    '${session}' == '${EMPTY}'
          Create API Session    admin_session
          ${session}=    Set Variable    admin_session
      END

      &{user_data}=   Create Dictionary    email=${email}    password=${password}    is_superuser=${is_superuser}
      ${response}=    POST On Session    ${session}    /api/users    json=${user_data}    expected_status=201

      ${user}=    Set Variable    ${response.json()}
      RETURN    ${user}
      
Create Random Test User
    [Documentation]    Create a test user with random email
    [Arguments]    ${session}    ${password}=test-password-123    ${is_superuser}=False

    ${random_id}=    Generate Random String    8    [LETTERS][NUMBERS]
    ${email}=        Set Variable    test-user-${random_id}@example.com

    ${user}=    Create Test User    ${session}    ${email}    ${password}    ${is_superuser}
    RETURN    ${user}

Delete Test User
    [Documentation]    Delete a test user (uses admin session)
    [Arguments]    ${session}    ${user_id}

    # Delete user
    ${response}=    DELETE On Session    ${session}    /api/users/${user_id}    expected_status=200
    RETURN    ${response.json()}

Get User Details
    [Documentation]    Get user details by ID
    [Arguments]    ${user_id}

    # Get admin session
    Create API Session    admin_session

    # Get user
    ${response}=    GET On Session    admin_session    /api/users/${user_id}    expected_status=200
    RETURN    ${response.json()}

Get Current User
    [Documentation]    Get current authenticated user details
    [Arguments]    ${session_alias}

    ${response}=    GET On Session    ${session_alias}    /users/me    expected_status=200
    RETURN    ${response.json()}

Get Admin User Details
    [Documentation]    Get current admin user details (session-based)
    [Arguments]    ${session_alias}

    ${user}=    Get Current User    ${session_alias}
    RETURN    ${user}

Get Admin User Details With Token
    [Documentation]    Get current admin user details using token (legacy compatibility)
    [Arguments]    ${token}

    Create Session    temp_user_session    ${API_URL}    verify=True
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    ${response}=    GET On Session    temp_user_session    /users/me    headers=${headers}    expected_status=200
    ${user}=        Set Variable    ${response.json()}
    Delete All Sessions
    RETURN    ${user}

List All Users
    [Documentation]    List all users (admin only)

    # Get admin session
    ${admin_session}=    Get Admin Session

    # Get users
    ${response}=    GET On Session    ${admin_session}    /api/users    expected_status=200
    RETURN    ${response.json()}

Update User
    [Documentation]    Update user details
    [Arguments]    ${user_id}    &{updates}

    # Get admin session
    ${admin_session}=    Get Admin Session

    # Update user
    ${response}=    PUT On Session    ${admin_session}    /api/users/${user_id}    json=${updates}    expected_status=200
    RETURN    ${response.json()}

Attempt User Login
    [Documentation]    Attempt to log in with user credentials
    [Arguments]    ${email}    ${password}

    ${session}=    Get User Session    ${email}    ${password}
    ${user}=       Get Current User    ${session}
    RETURN    ${user}

Attempt User Login With Invalid Credentials
    [Documentation]    Attempt login with invalid credentials and return response
    [Arguments]    ${email}    ${password}

    Create Session    temp    ${API_URL}    verify=True
    &{auth_data}=    Create Dictionary    username=${email}    password=${password}
    &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=    POST On Session    temp    /auth/jwt/login    data=${auth_data}    expected_status=any
    Delete All Sessions
    RETURN    ${response}

Cleanup Test User
    [Documentation]    Create and cleanup a test user (for use in test teardown)
    [Arguments]    ${user_email}

    TRY
        # Try to find and delete the user
        ${users}=    List All Users
        FOR    ${user}    IN    @{users}
            IF    "${user}[email]" == "${user_email}"
                Delete Test User    ${user}[user_id]
                Log    Deleted test user: ${user_email}    INFO
                RETURN
            END
        END
        Log    Test user not found: ${user_email}    INFO
    EXCEPT    AS    ${error}
        Log    Failed to cleanup test user ${user_email}: ${error}    WARN
    END

