*** Settings ***
Library          RequestsLibrary
Library          Collections
Library          OperatingSystem
Variables        ../test_env.py

*** Keywords ***

Create Basic Auth Header
    [Documentation]
    # Concatenate the username and password in "user:pass" format
    [Arguments]    ${username}    ${password}
    ${credentials}=    Set Variable    ${username}:${password}
    Log To Console    Credentials: ${credentials}
 
    # Encode the credentials using base64 encoding
    ${encoded}=    Evaluate    str(base64.b64encode('${credentials}'.encode('utf-8')), 'utf-8')    modules=base64
    Log To Console    Encoded Credentials: ${encoded}
 
    # Create a headers dictionary and add the Authorization header
    ${headers}=    Create Dictionary    Content-Type=application/json
    Set To Dictionary    ${headers}    Authorization=Basic ${encoded}
    RETURN    ${headers}


