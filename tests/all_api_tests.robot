*** Settings ***
Documentation    Master Test Suite for All Friend-Lite API Endpoints
Suite Setup      Master Suite Setup
Suite Teardown   Master Suite Teardown

*** Test Cases ***

Run Auth Tests
    [Documentation]    Execute authentication and user management tests
    [Tags]             auth    users    suite
    Run Tests    auth_tests.robot

Run Memory Tests
    [Documentation]    Execute memory management tests
    [Tags]             memory    suite
    Run Tests    memory_tests.robot

Run Conversation Tests
    [Documentation]    Execute conversation management tests
    [Tags]             conversation    suite
    Run Tests    conversation_tests.robot

Run Health Tests
    [Documentation]    Execute health and status tests
    [Tags]             health    status    suite
    Run Tests    health_tests.robot

Run Chat Tests
    [Documentation]    Execute chat service tests
    [Tags]             chat    suite
    Run Tests    chat_tests.robot

Run Client Queue Tests
    [Documentation]    Execute client and queue management tests
    [Tags]             client    queue    suite
    Run Tests    client_queue_tests.robot

Run System Admin Tests
    [Documentation]    Execute system and admin tests
    [Tags]             system    admin    suite
    Run Tests    system_admin_tests.robot

*** Keywords ***

Master Suite Setup
    [Documentation]    Setup for the entire test suite
    Log    Starting Friend-Lite API Test Suite
    Log    Testing against: ${API_URL}

Master Suite Teardown
    [Documentation]    Cleanup for the entire test suite
    Log    Friend-Lite API Test Suite completed

Run Tests
    [Documentation]    Run a specific test file
    [Arguments]    ${test_file}
    Log    Executing: ${test_file}
    # Note: This is a placeholder - actual test execution handled by robot command