*** Settings ***
Documentation    Debug Pipeline Step by Step
Resource         ../resources/integration_keywords.robot
Resource         ../resources/setup_resources.robot
Suite Setup      Suite Setup
Suite Teardown   Suite Teardown 

*** Test Cases ***

Test server connection
    [Documentation]    Test connection to the server
    [Tags]             debug    connection    todo

    Log    Testing server connection    INFO
    Fail    Test not written yet - placeholder test

Login to server
    [Documentation]    Test logging in to the server from mobile client
    Log    Logging in to server    INFO
    Fail    Test not written yet - placeholder test

Scan bluetooth devices
    [Documentation]    Scan for available bluetooth devices
    Log    Scanning bluetooth devices    INFO
    Fail    Test not written yet - placeholder test

Filter devices by omi
    [Documentation]    Filter scanned devices by omi
    Log    Filtering devices by omi    INFO
    Fail    Test not written yet - placeholder test

Connect to bluetooth device
    [Documentation]    Connect to a bluetooth device
    Log    Connecting to bluetooth device    INFO
    Fail    Test not written yet - placeholder test

Get device codec
    [Documentation]    Get the codec information from the device
    Log    Getting device codec    INFO
    Fail    Test not written yet - placeholder test

Get device battery level
    [Documentation]    Get the battery level from the device
    Log    Getting device battery level    INFO
    Fail    Test not written yet - placeholder test

Start audio stream
    [Documentation]    Start streaming audio from the device
    Log    Starting audio stream    INFO
    Fail    Test not written yet - placeholder test
