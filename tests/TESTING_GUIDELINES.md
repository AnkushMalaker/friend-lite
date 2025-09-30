# Robot Framework Testing Guidelines

This file provides specific guidelines for organizing and writing Robot Framework tests in this project.

## Test Organization Principles

### Resource File Organization

Each resource file should have a clear purpose and contain related keywords. Resource files should include documentation explaining what types of functions belong in that file.

#### Resource File Categories

**setup_resources.robot**
- Docker service management (start/stop services)
- Environment validation
- Health checks and service dependency verification
- System preparation keywords
- Any keywords that prepare the testing environment

**session_resources.robot**
- API session creation and management
- Authentication workflows
- Token management (when needed for external tools like curl)
- Session validation and cleanup
- Keywords that handle API authentication and session state

**user_resources.robot**
- User account creation, deletion, and management
- User-related operations and utilities
- User permission validation
- Keywords specific to user account lifecycle

**integration_keywords.robot**
- Core integration workflow keywords
- File processing and upload operations
- System interaction keywords that don't fit in other categories
- Complex multi-step operations that combine multiple services

### Verification vs Setup Separation

**Verification Steps**
- **MUST be written directly in test files, not abstracted into resource keywords**
- Keep verifications close to the test logic for readability and maintainability
- Use descriptive assertion messages that explain what is being verified
- Example: `Should Be Equal As Integers    ${response.status_code}    200    Health check should return 200`
- Verification keywords should only exist in resource files if they perform complex multi-step verification that needs to be reused across multiple test suites

**Setup/Action Keywords**
- Environment setup, service management, and system actions belong in resource files
- These can be reused across multiple tests and suites
- Focus on "what to do" rather than "what to verify"
- Examples: `Get Admin API Session`, `Upload Audio File For Processing`, `Start Docker Services`

**Suite-Level Keywords**
- If a specific set of verifications needs to be repeated multiple times within a single test suite, create keywords at the suite level (in the *** Keywords *** section of the test file)
- These should be specific to that suite's testing needs
- Only create suite-level keywords when the same verification logic is used 3+ times in the same suite

## Code Style Guidelines

### Human Readability
- Tests should be readable by domain experts without deep Robot Framework knowledge
- Use descriptive keyword names that explain the business purpose
- Prefer explicit over implicit - make test intentions clear
- Use meaningful variable names and comments where helpful
- Avoid Robot Framework-specific jargon in test names and documentation

### Test Structure
```robot
*** Test Cases ***
Test Name Should Describe Business Scenario
    [Documentation]    Clear explanation of what this test validates
    [Tags]            relevant    tags

    # Arrange - Setup test data and environment
    ${session}=    Get Admin API Session

    # Act - Perform the operation being tested
    ${result}=    Upload Audio File For Processing    ${session}    ${TEST_FILE}

    # Assert - Verify results directly in test (NOT in resource keywords)
    Should Be True    ${result}[successful] > 0    At least one file should be processed successfully
    Should Contain    ${result}[message]    processing completed    Processing should complete successfully
```

### Resource File Documentation
Each resource file should start with clear documentation:

```robot
*** Settings ***
Documentation    Brief description of this resource file's purpose
...
...              This file contains keywords for [specific purpose].
...              Keywords in this file should handle [what types of operations].
...
...              Examples of keywords that belong here:
...              - Keyword type 1
...              - Keyword type 2
...
...              Keywords that should NOT be in this file:
...              - Verification/assertion keywords (belong in tests)
...              - Keywords specific to other domains
```

### Authentication Pattern
- Tests should use session-based authentication via `session_resources.robot`
- Avoid passing tokens directly in tests - use sessions instead
- Extract tokens from sessions only when required for external tools (like curl)

Example:
```robot
# Good - Session-based approach
${admin_session}=    Get Admin API Session
${conversations}=    Get User Conversations    ${admin_session}

# Avoid - Direct token handling in tests
${token}=    Get Admin Token
${conversations}=    Get User Conversations    ${token}
```

## File Naming and Structure

### Test Files
- Use descriptive names that indicate the testing scope
- Example: `full_pipeline_test.robot`, `user_management_test.robot`
- Use `_test.robot` suffix for test files

### Resource Files
- Use `_resources.robot` suffix
- Name should indicate the domain: `session_resources.robot`, `user_resources.robot`

### Keywords
- Use descriptive names with clear action words
- Start with action verb when possible: `Get User Conversations`, `Upload Audio File`, `Create Test User`
- Avoid abbreviations unless they're widely understood in the domain
- Use consistent naming patterns across similar keywords

## Error Handling

### Resource Keywords
- Should handle expected error conditions gracefully
- Use appropriate Robot Framework error handling (TRY/EXCEPT blocks)
- Log meaningful error messages for debugging
- Fail fast with clear error messages when setup fails

### Test Assertions
- Write verification steps directly in tests with clear failure messages
- Use descriptive assertion messages that explain what went wrong
- Example: `Should Be Equal    ${status}    active    User should be in active status after creation`
- Include relevant context in failure messages (expected vs actual values)

## Keywords vs Inline Code

### When to Create Keywords
- Reusable operations that are used across multiple tests or suites
- Complex multi-step setup or teardown operations
- Operations that encapsulate business logic or domain concepts
- Operations that interact with external systems (APIs, databases, files)

### When to Keep Code Inline
- Verification steps (assertions) - these should almost always be inline in tests
- Simple operations that are only used once
- Test-specific logic that doesn't need to be reused
- Variable assignments and simple data manipulation

## Variable and Data Management

### Test Data
- Use meaningful variable names that describe the data's purpose
- Define test data at the appropriate scope (suite variables for shared data, test variables for test-specific data)
- Store complex test data in separate variable files when it becomes large
- Use descriptive names: `${VALID_USER_EMAIL}` instead of `${EMAIL1}`

### Environment Configuration
- Load environment variables through `test_env.py`
- Use consistent variable naming across tests
- Document required environment variables and their purposes

## Future Additions

As we develop more conventions and encounter new patterns, we will add them to this file:
- Performance testing guidelines
- Data management patterns
- Mock and test double strategies
- Continuous integration considerations
- Test reporting and metrics
- Parallel test execution patterns
- Test data isolation strategies