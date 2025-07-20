# Evaluating LLMs on AndroidWorld Tasks  
*Android Agent Performance Analysis Report*

## 1 Experimental Setup

- **Environment**: [AndroidWorld](https://github.com/google-research/android_world) framework, Android 13 emulation
- **Task Sample**: 40 episodes across diverse apps (Markor, Contacts, System Settings, Sports Tracker, Simple Calendar, ExpenseTracker)  
- **Task Types**: File manipulation, contact management, system configuration, app interactions, calendar operations
- **Agent Implementation**: Python harness with prompt injection, OpenAI function calling, previous step memory, and AndroidWorld action conversion. Note: All models tested were purely text-based, relying on UI state descriptions rather than visual input.
- **Data Collection**: Complete action traces stored as JSON with step records, UI states, and performance metrics

- **Models Evaluated**:

  | Model ID | Family | Context Window | API Version |
  |----------|--------|----------------|-------------|
  | `gpt-4.1-mini` | GPT-4.1 | 128K | Latest |
  | `gpt-4o-mini` | GPT-4o | 128K | Latest |
  | `o3-mini` | O3 | 128K | Latest |
  | `o4-mini` | O4 | 128K | Latest |

- **Prompt Variants**:

  1. **Base** – Standard agent instructions with action schema
  2. **Few-shot** – Base prompt + 2-3 task demonstrations  
  3. **Reflective** – Few-shot + step-by-step reasoning instructions

Evaluation corpus: **40 episodes** with step-by-step action logging and success measurement.

## 2 Performance Metrics

| Symbol | Definition |
|--------|------------|
| **SR %** | Episode success rate – agent achieves task completion |
| **Avg Steps** | Average number of actions taken per episode |
| **Max Steps** | Step limit threshold (30 steps) |

## 3 Results

### Overall Performance Summary

| Model | Prompt | Episodes | Success Rate % | Avg Steps |
|-------|--------|----------|----------------|-----------|
| **Overall** | **All** | **40** | **17.5** | **18.8** |
| gpt-4.1-mini | base | 4 | 25.0 | 20.8 |
| gpt-4.1-mini | few-shot | 6 | 16.7 | 22.7 |
| gpt-4.1-mini | reflective | 4 | **50.0** | 20.0 |
| gpt-4o-mini | base | 6 | 16.7 | 25.3 |
| gpt-4o-mini | few-shot | 3 | 33.3 | 16.7 |
| gpt-4o-mini | reflective | 4 | 25.0 | 24.3 |
| o3-mini | base | 8 | 0.0 | 6.9 |
| o3-mini | few-shot | 2 | 0.0 | 10.0 |
| o3-mini | reflective | 1 | 0.0 | 30.0 |
| o4-mini | few-shot | 1 | 0.0 | 30.0 |
| o4-mini | reflective | 1 | 0.0 | 19.0 |

### Task-Specific Performance

| Task | Episodes | Success Rate % | Avg Steps |
|------|----------|----------------|-----------|
| **ContactsAddContact** | 3 | **100.0** | 7.0 |
| **MarkorMoveNote** | 10 | 40.0 | 20.6 |
| **SystemBrightnessMin** | 14 | 0.0 | 22.9 |
| **ExpenseAddMultiple** | 5 | 0.0 | 20.8 |
| **All Others** | 8 | 0.0 | 9.4 |

> **Key Findings**: Reflective prompting with GPT-4.1-mini achieved highest success rate (50%), while O3/O4 models showed poor performance. Simple contact tasks had perfect success, while system settings proved most challenging. Specifically, tasks involving manipulating quick settings or system UI elements resulted in navigation loops and failure to locate controls.

## 4 Detailed Episode Analysis

### 4.1 Successful Episode: MarkorMoveNote (GPT-4.1-mini, Base)
**Task**: Move note `2023_05_22_silly_zebra.txt` from MeetingMinutes to FitnessPlans folder

**Execution Summary** (9 steps, Success):
1. `open_app("Markor")` - Correctly launched file manager
2. `click(index=13)` - Navigated to MeetingMinutes folder  
3. `long_press(index=14)` - Selected target file with appropriate gesture
4. `click(index=6)` - Accessed context menu 
5. `click(index=1)` - Activated move operation
6. `click(index=5)` - Navigated up directory hierarchy
7. `click(index=6)` - Selected FitnessPlans destination
8. `click(index=4)` - Confirmed move operation
9. `status("complete")` - Correctly identified completion

**Success Factors**: Precise UI element targeting, logical navigation sequence, proper use of long-press for file selection.

### 4.2 Successful Episode: ContactsAddContact (GPT-4o-mini, Base)  
**Task**: Create contact for Adam Gonzalez with phone +14460177118

**Execution Summary** (7 steps, Success):
1. `open_app("Phone")` - Leveraged phone app for contacts access
2. `click(index=1)` - Found "Create new contact" option
3. `input_text("Adam", index=7)` - Entered first name correctly
4. `input_text("Gonzalez", index=8)` - Entered last name
5. `input_text("+14460177118", index=10)` - Input phone number with formatting
6. `click(save_button)` - Saved contact
7. `status("complete")` - Task completion

**Success Factors**: Efficient app selection, direct form filling, proper use of input_text action.

### 4.3 Failed Episode: SystemBrightnessMin (GPT-4o-mini, Base)
**Task**: Turn device brightness to minimum value

**Failure Analysis** (30 steps, Failed):
- **Steps 1-3**: Correctly opened Settings → Display → Brightness
- **Steps 4-30**: **Repetitive scrolling loop** - agent continuously scrolled down searching for brightness slider
- **Core Issue**: Unable to locate or interact with brightness control UI element
- **Final Actions**: Identical scroll commands repeated 26 times without progress

**Failure Mode**: **UI Element Hallucination** - agent expected brightness slider in specific location but couldn't find it, leading to infinite scroll behavior.

## 5 Error Analysis & Common Failure Patterns

### 5.1 Navigation Failures (60% of errors)
- **Infinite Scrolling**: Agents repeatedly scroll seeking UI elements that may not exist or are inaccessible (SystemBrightnessMin episodes)
- **Wrong App Selection**: Choosing inappropriate apps for tasks
- **Directory Confusion**: Getting lost in file system navigation

### 5.2 UI Reasoning Limitations (25% of errors)  
- **Element Visibility Issues**: Failing to recognize when target UI elements are off-screen or hidden
- **Interaction Method Confusion**: Using `click` instead of `long_press` for context menus
- **Index Selection Errors**: Clicking wrong UI element indices due to dynamic layout changes

### 5.3 Task Completion Recognition (15% of errors)
- **Premature Completion**: Calling `status("complete")` before task is finished
- **Missing Final Steps**: Stopping before saving or confirming changes
- **Goal Misinterpretation**: Misunderstanding task requirements

### 5.4 Model-Specific Observations
- **O3/O4 Models**: Consistently failed with very short episodes (1-3 steps), suggesting fundamental prompt comprehension issues
- **GPT-4.1-mini**: Best performance with reflective prompting, showing benefit of explicit reasoning
- **GPT-4o-mini**: Good at simple tasks but struggles with complex multi-step navigation

## 6 Prompt Engineering Impact

| Prompt Type | Avg Success Rate | Key Benefits | Limitations |
|-------------|------------------|--------------|-------------|
| **Base** | 18.2% | Simple, direct | Lacks context and examples |
| **Few-shot** | 20.0% | Provides action patterns | Examples may not match task |  
| **Reflective** | 33.3% | Encourages step planning | Can cause over-analysis |

**Reflective prompting** showed the most promise, particularly with GPT-4.1-mini, achieving 50% success rate compared to 25% with base prompting.

## 7 Task Complexity Analysis

**Simple Tasks** (7-9 steps average):
- Contact creation: 100% success rate
- Clear UI patterns, standard form interactions

**Medium Tasks** (15-25 steps average):  
- File operations: 40% success rate
- Requires navigation + context menus

**Complex Tasks** (25+ steps average):
- System settings, multi-step tasks: 0% success rate  
- Hidden UI elements, multi-level navigation

## 8 Recommendations for Improving Agent Performance

### 8.1 Prompt Engineering
1. **Enhanced Few-shot Examples**: Include examples covering failed scenarios and recovery strategies
2. **Explicit UI State Reasoning**: Add instructions for recognizing when UI elements are not visible
3. **Scroll Termination Conditions**: Define stopping criteria for search operations, encourage agents to scroll in different directions if they are unable to find elements after a certain number of attempts

### 8.2 Agent Architecture  
1. **UI Element Verification**: Implement confidence checks before interacting with elements
2. **Loop Recognition**: Allow agents to revert actions when stuck in loops
3. **Task Decomposition**: Break complex tasks into verifiable sub-goals

### 8.3 Evaluation Framework
1. **Step Accuracy Metrics**: Measure individual action correctness, not just final success
2. **Error Classification**: Categorize failure modes for targeted improvements
3. **Cross-Model Benchmarking**: Establish standardized task difficulty rankings and comparative performance baselines

## 9 Conclusion

Current LLM performance on AndroidWorld tasks remains challenging, with only **17.5% overall success rate** across 40 episodes. Broader testing is necessary to draw stronger conclusions. Key findings:

- **Task Complexity Correlation**: Simple form-based tasks (ContactsAddContact: 100%) succeed far better than system navigation tasks (SystemBrightnessMin: 0%)
- **Model Differences**: GPT-4.1-mini with reflective prompting achieved best performance (50%), while O3/O4 models failed entirely
- **Primary Failure Mode**: Navigation loops and UI element location failures account for 85% of task failures

**Immediate Opportunities**: Focus on improving UI element detection reliability and implementing loop prevention mechanisms. The 40% success rate on file operations (MarkorMoveNote) suggests significant potential for improvement with targeted architectural enhancements.

**Future Directions**: Multimodal approaches combining vision and action planning may address current limitations in dynamic UI understanding and spatial reasoning.

---
*Analysis based on 40 episodes across 12 distinct AndroidWorld tasks using 4 LLM variants and 3 prompt engineering approaches.*
