## ADDED Requirements

### Requirement: C++ Ahead-Of-Time (AOT) Kernel Registry
The system SHALL provide a thread-safe C++ Kernel Registry (`bsm_rli::KernelRegistry`) that manages AOT VRAM/CPU function pointers and maps them to EBNF schema signatures.

#### Scenario: Registering a new C++ micro-kernel
- **WHEN** a developer registers a C++ function pointer with a unique kernel identifier and parameter signature
- **THEN** the registry SHALL store the function pointer in VRAM/host RAM and expose its signature for EBNF grammar generation

#### Scenario: Dispatching execution to a registered kernel
- **WHEN** an intercepted argument payload matches a registered kernel identifier
- **THEN** the registry SHALL invoke the corresponding function pointer in < 5 microseconds and return the serialized result string

### Requirement: EBNF Logit Grammar Generation
The system SHALL dynamically compile the set of active registered kernels into an EBNF grammar specification suitable for `llguidance` and `XGrammar` engines.

#### Scenario: Compiling active registry into EBNF schema
- **WHEN** `KernelRegistry::generate_ebnf()` is invoked
- **THEN** the system SHALL output a valid EBNF grammar constraining next-token generation strictly to `<|jit_start|>` followed by valid kernel calls and parameters
