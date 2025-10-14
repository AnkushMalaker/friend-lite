*** Settings ***
Documentation    Authentication and User Management API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Resource         ../resources/setup_resources.robot
Resource         ../resources/user_resources.robot
Suite Setup      Suite Setup
Test Setup       Clear Test Databases
Suite Teardown   Suite Teardown

*** Variables ***
# Test users are now imported from test_env.py via resource files

*** Test Cases ***

Login With Valid Credentials Test
    [Documentation]    Test successful login with admin credentials
    [Tags]             auth    login    positive
    ${user}=           Get Admin User Details    api
    Should Be Equal    ${user}[email]    ${ADMIN_EMAIL}

Login With Invalid Credentials Test
    [Documentation]    Test login failure with invalid credentials
    [Tags]             auth    login    negative
    Get Anonymous Session    anon_session

    &{auth_data}=    Create Dictionary    username=${ADMIN_EMAIL}    password=wrong-password
    &{headers}=      Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=    POST On Session    anon_session    /auth/jwt/login    data=${auth_data}    headers=${headers}    expected_status=400
    Should Be Equal As Integers    ${response.status_code}    400

Get Current User Test
    [Documentation]    Test getting current authenticated user
    [Tags]             auth    user    positive

    Create API Session    api
    ${user}=           Get Admin User Details   api

    Dictionary Should Contain Key    ${user}    email
    Dictionary Should Contain Key    ${user}    id
    Should Be Equal       ${user}[email]    ${ADMIN_EMAIL}

Unauthorized Access Test
    [Documentation]    Test that endpoints require authentication
    [Tags]             auth    security    negative
    Get Anonymous Session    anon_session

    # Try to access protected endpoint without token
    ${response}=    GET On Session   anon_session   /users/me    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Create User Test
    [Documentation]    Test creating a new user (admin only)
    [Tags]             users    admin    positive

    Create API Session    api
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${user}=           Create Test User    api    ${test_email}    ${TEST_USER_PASSWORD}

    Should Be Equal    ${user}[user_email]    ${test_email}
    Should Contain     ${user}[message]    created successfully

    # Cleanup
    Delete Test User    api    ${user}[user_id]

Create Admin User Test
    [Documentation]    Test creating an admin user
    [Tags]             users    admin    positive


    Create API Session    session
    ${test_admin_email}=    Set Variable    test-admin-${RANDOM_ID}@example.com
    ${user}=           Create Test User    session     ${test_admin_email}    ${TEST_USER_PASSWORD}    ${True}

    Should Be Equal    ${user}[user_email]    ${test_admin_email}
    Should Contain     ${user}[message]    created successfully

    # Cleanup
    Delete Test User   session    ${user}[user_id]

Get All Users Test
    [Documentation]    Test getting all users (admin only)
    [Tags]             users    admin    positive
    Create API Session    api
    ${response}=       GET On Session    api    /api/users

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True     isinstance($response.json(), list)

    # Should contain at least the admin user
    ${users}=    Set Variable    ${response.json()}
    ${admin_found}=    Set Variable    ${False}
    FOR    ${user}    IN    @{users}
        IF    '${user}[email]' == '${ADMIN_EMAIL}'
            ${admin_found}=    Set Variable    ${True}
        END
    END
    Should Be True    ${admin_found}

Non-Admin User Cannot Create Users Test
    [Documentation]    Test that non-admin users cannot create users
    [Tags]             users    security    negative
    ${random_id}=    Generate Random String    8    [LETTERS][NUMBERS]
    # Create a non-admin user
     # Create user
    Create API Session    user_session    email=${TEST_USER_EMAIL}    password=${TEST_USER_PASSWORD}
    &{user_data}=   Create Dictionary    email=${random_id}${TEST_USER_EMAIL}    password=${TEST_USER_PASSWORD}    is_superuser=${False}
    ${response}=    POST On Session    user_session    /api/users    json=${user_data}    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

Update User Test
    [Documentation]    Test updating a user (admin only)
    [Tags]             users    admin    positive

    Create API Session    session
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${user}=           Create Test User   session    ${test_email}    ${TEST_USER_PASSWORD}

    # Update user to admin
    &{update_data}=    Create Dictionary    email=${test_email}    password=${TEST_USER_PASSWORD}    is_superuser=${True}

    ${response}=       PUT On Session    session    /api/users/${user}[user_id]    json=${update_data}
    Should Be Equal As Integers    ${response.status_code}    200

    ${updated_user}=   Set Variable    ${response.json()}
    Should Be True     ${updated_user}[is_superuser]

Delete User Test
    [Documentation]    Test deleting a user (admin only)
    [Tags]             users    admin    positive

    Create API Session    session
    ${test_email}=     Set Variable    test-user-${RANDOM_ID}@example.com
    ${user}=           Create Test User    session     ${test_email}    ${TEST_USER_PASSWORD}

    # Delete the user
    Delete Test User    session   ${user}[user_id]

    # Verify user is deleted by trying to login
    &{auth_data}=      Create Dictionary    username=${test_email}    password=${TEST_USER_PASSWORD}
    &{headers}=        Create Dictionary    Content-Type=application/x-www-form-urlencoded

    ${response}=       POST On Session    session    /auth/jwt/login    data=${auth_data}    headers=${headers}    expected_status=400
    Should Be Equal As Integers    ${response.status_code}    400

