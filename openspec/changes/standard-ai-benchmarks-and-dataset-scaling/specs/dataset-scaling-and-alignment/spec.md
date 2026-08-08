## ADDED Requirements

### Requirement: 50,000+ Sample Synthetic Dataset Scaling and Alignment
The system SHALL provide a dataset generator (`dataset/generate_synthetic_data.py`) producing 50,000+ synthetic micro-kernel instruction pairs across 30+ primitives in 6 operational domains, mixed with 10,000 conversational rejection samples to prevent over-triggering.

#### Scenario: Generating 50,000+ training dataset
- **WHEN** `dataset/generate_synthetic_data.py --scale 50000` is executed
- **THEN** it SHALL write 50,000+ training samples to `dataset/bsm_rli_sft_50k.json`
