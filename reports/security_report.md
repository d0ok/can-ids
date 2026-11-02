# Automotive CAN Intrusion Detection: Security Report and Threat Model

**Project:** ML anomaly detection on CAN traffic (HCRL Car-Hacking dataset)
**Author:** Dhia Benothman  |  **Date:** November 2026
**Code and figures:** see `src/` and `reports/figures/`

## 1. Summary

I built an unsupervised intrusion detector for CAN bus traffic. All models are trained on **normal traffic only**, and attack files are used only for testing. The final design is a **hybrid**: a CAN-ID whitelist rule plus an Isolation Forest on per-ID frequency and timing features.

| Attack | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| DoS | 0.928 | 1.000 | 0.963 | 1.5% |
| Fuzzy | 0.938 | 0.992 | 0.964 | 1.0% |
| Gear spoofing | 0.832 | 0.981 | 0.900 | 3.1% |
| RPM spoofing | 0.829 | 0.989 | 0.902 | 3.4% |

*(Hybrid detector, full files, message-level labels. FPR is measured on normal messages inside attack files.)*

Key findings:
1. A first-pass Isolation Forest with naive timing features **failed** (DoS F1 0.0002, Gear 0.03). It flagged legitimate messages delayed by bus contention, not injected ones. Directional per-ID features fixed Gear and RPM (F1 about 0.90).
2. **DoS and Fuzzy are solved almost entirely by the whitelist rule**, because the dataset's attackers inject CAN IDs that never occur in normal traffic. This is a property of the dataset, not a general result (see Section 5).
3. An **LSTM next-ID predictor underperformed** the forest hybrid on every attack (F1 0.62-0.75, FPR 16-26%).
4. Most false positives of the forest hybrid occur **within 20 messages of an injection**. Far from attacks its FPR matches the calibration target of 0.5%.

## 2. System and assumptions

- **Target:** a vehicle CAN bus (the dataset was logged through the OBD-II port of a real car).
- **Protocol weaknesses:** CAN frames carry no sender authentication or encryption, and any node can transmit any ID. Arbitration favors low IDs, so a node sending ID `0x000` wins the bus.
- **IDS placement:** a passive listener on the bus. It sees all frames, with timestamps, but cannot block them.
- **Assumption:** a clean recording of normal operation is available for training.

## 3. Threat model

**Assets:** safety-critical ECU functions (powertrain, gear, instrument cluster), and bus availability.

**Attacker capabilities** (as exercised by the dataset): physical or remote access to the bus (OBD-II port or a compromised ECU), ability to inject arbitrary frames at high rate, and knowledge of which CAN IDs carry which signals.

| Dataset attack | Behavior | Threat category | Impact |
|---|---|---|---|
| DoS | ID `0x000` injected every 0.3 ms | Denial of service | Legitimate traffic starved by arbitration |
| Fuzzy | Random IDs and payloads every 0.5 ms | Tampering / unintended behavior | Unpredictable ECU reactions |
| Gear spoofing | Gear-related ID injected every 1 ms | Spoofing | False gear information |
| RPM spoofing | RPM-related ID injected every 1 ms | Spoofing | False RPM display or control input |

**Not covered by this dataset or my evaluation:** replay attacks, ECU suspension or bus-off attacks, masquerade attacks (silencing the real ECU and impersonating it), payload-only manipulation at normal rates, and adaptive attackers who know the detector.

## 4. Method

- **Data:** four attack files plus one normal run (988,871 messages, 27 unique IDs). Injected messages are labelled `T`.
- **Split:** the first 70% of the normal run is training data, the remaining 30% is held out to calibrate thresholds (99.5th percentile of normal scores). No shuffling, since CAN is a time series.
- **Features (causal, using only current and past messages):** inter-arrival time (global and per-ID), per-ID message frequency in a 100-message window, DLC and DLC window statistics. Version 2 uses a per-ID frequency ratio (observed vs expected rate), a one-sided "too fast" timing feature, and a DLC-mismatch flag.
- **Models:** Isolation Forest (200 trees); LSTM (64 units, 20-message window, predicts the next CAN ID, with surprise as the anomaly score); whitelist rule (flag any ID unseen in training).
- **Metrics:** precision, recall, F1, ROC-AUC and FPR per attack type.

## 5. Results and analysis

### 5.1 Detector comparison (F1)

| Attack | IF v1 (naive) | IF v2 | IF v2 + rule | Rule only | LSTM + rule* |
|---|---|---|---|---|---|
| DoS | 0.000 | 0.000 | 0.963 | 1.000 | 0.754 |
| Fuzzy | 0.758 | 0.010 | 0.964 | 0.993 | 0.675 |
| Gear | 0.033 | 0.900 | 0.900 | 0.000 | 0.616 |
| RPM | 0.022 | 0.902 | 0.902 | 0.000 | 0.661 |

*\*LSTM evaluated on a contiguous 1M-row slice per file because windows need contiguous messages.*

Figures: `figures/f1_by_detector.png`, `figures/roc_comparison.png`, `figures/confusion_hybrid.png`.

### 5.2 Why the first model failed
Naive timing features made the forest flag legitimate messages delayed by bus contention (inter-arrival ratio of 9.6 on DoS), while injected messages looked normal on most features. Only 0.05% of its DoS flags were true attacks. Single-feature analysis showed that per-ID frequency alone separated the attacks (AUC 0.998 on DoS, 0.991 on Gear). The fix was to make features **directional and per-ID**.

### 5.3 The whitelist dominates DoS and Fuzzy
The rule alone scores F1 1.000 on DoS and 0.993 on Fuzzy. The forest adds false positives there (precision falls from 1.000 to 0.928 on DoS) and gains only 0.5 points of recall on Fuzzy. The forest is needed for Gear and RPM, where injected IDs are legitimate and the rule has recall 0.

### 5.4 False positives cluster around attacks
Forest-hybrid FPR on normal messages, by distance to the nearest injection (1M-row slices):

| Attack | 1-20 msgs | 21-200 | 201-2000 | >2000 |
|---|---|---|---|---|
| DoS | 6.2% | 0.9% | 0.6% | 0.5% |
| Fuzzy | 2.0% | 1.1% | 0.5% | 0.5% |
| Gear | 8.2% | 1.1% | 0.5% | 0.5% |
| RPM | 8.8% | 1.2% | 0.5% | 0.5% |

In quiet periods the false-positive rate matches the 0.5% calibration target. Messages immediately around an injection burst are hard to separate from the attack itself. See `figures/fpr_vs_distance.png`.

### 5.5 LSTM result
The LSTM catches nearly all attacks (recall 0.97-1.00) but flags 11% of normal messages even far from any injection, and 29-44% within 20 messages of one. The second effect is **context contamination**: injected messages in the 20-message input window make following legitimate messages look unpredictable. The 11% floor is not explained by contamination. My hypothesis, **untested**, is that the model depends on message ordering and raw timing, which differ between capture sessions, whereas the forest's per-ID frequency ratios do not. At a matched 5% FPR, forest recall on Gear and RPM is 0.999, against 0.74 and 0.15 for the LSTM. This result applies to this configuration (small LSTM, 400k training windows, 6 epochs), not to sequence models in general.

## 6. Limitations

1. **Dataset attacks are easy for a whitelist.** DoS and Fuzzy use IDs absent from normal traffic. An attacker who injects on a legitimate ID (for example a flood on a real high-priority ID) would depend on the forest's frequency features, the weaker layer.
2. **Payload is mostly ignored.** The features use frequency, timing and DLC, not payload values. A spoofer that sends wrong data at the legitimate rate and timing would be invisible. This was not tested.
3. **Rate-matched spoofing was not tested.** Gear and RPM injections are detectable largely because they arrive about 7x faster than the legitimate signal.
4. **Feature design used attack-file diagnostics.** Thresholds were calibrated on normal data only, but I chose the version-2 features after examining attack-file behavior. Reported scores are therefore somewhat optimistic compared with a fully blind design.
5. **Message-level labels penalize detection inside bursts.** Legitimate messages adjacent to injections score as false positives, which inflates FPR relative to what a burst-level alert would show.
6. **Single vehicle, single recording session.** Generalization across vehicles, drivers and capture sessions is unknown.
7. **The LSTM comparison covers one configuration** and a 1M-row slice per file.

## 7. Recommendations

- **Layer the defenses:** whitelist for unknown IDs, frequency model for known IDs, and a payload-range model for signal-level spoofing.
- **Aggregate alerts per burst** (for example, k alarms within 100 ms) so the 3-4% message-level FPR does not become dozens of alarms per second at about 1,950 messages/s.
- **Train on multiple capture sessions and vehicles** and report cross-session results.
- **Evaluate adaptive attacks:** injection on legitimate IDs, rate-matched spoofing, and payload-only manipulation.
- **Complement detection with prevention** (message authentication such as SecOC, gateway filtering). An IDS on a bus without authentication detects attacks but cannot stop them.

## 8. Reproducibility

Pipeline order: `src/load_data.py` → `src/features.py` → `src/train_iforest.py` → `src/train_iforest_v2.py` → `src/eval_hybrid.py` → `src/train_lstm.py` → `src/compare_lstm.py` → `src/fpr_by_distance.py` → `src/make_figures.py`. Random seed 42 throughout. Data is not included in the repository (see README).

## 9. Dataset citation

This work uses the HCRL Car-Hacking dataset (Hacking and Countermeasure
Research Lab, Korea University). As requested by the dataset authors:

- Song, H. M., Woo, J., & Kim, H. K. (2020). In-vehicle network intrusion
  detection using deep convolutional neural network. *Vehicular
  Communications*, 21, 100198.
- Seo, E., Song, H. M., & Kim, H. K. (2018). GIDS: GAN based Intrusion
  Detection System for In-Vehicle Network. *2018 16th Annual Conference on
  Privacy, Security and Trust (PST)*. IEEE.
