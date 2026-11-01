# Phase 15 — Ολοκλήρωση πιλοτικής αξιολόγησης (ΟΔΗΓΙΕΣ)

**Σημαντικό:** το πακέτο είναι *workflow* και δεν περιλαμβάνει αποτελέσματα 78 πραγματικών LLM κλήσεων. Τα δύο ήδη εκτελεσμένα development-pilot παραδείγματα **δεν επαναλαμβάνονται** ούτε περιλαμβάνονται στην αξιολόγηση των 78. Οι 40 υποψήφιες αρνητικές ερωτήσεις έχουν μόνο author/BM25/E5 screening, όχι ανεξάρτητο gold label.

## Τοποθέτηση

Αντέγραψε το `scripts/phase15_complete_workflow.py` στο `scripts/` του repository και το `tests/test_phase15_complete_workflow.py` στον φάκελο `tests/`. Το αρχείο οδηγιών μπορεί να μπει στο `docs/`.

**Μη διαγράψεις ή αντικαταστήσεις** `results/phase15_local/cases.json`, τα προηγούμενα 80-case αρχεία ή το `phase15_pilot2_predictions.json`. Δεν απαιτείται να εγκαταστήσεις νέο Python dependency.

## 1. Prepare (offline, ασφαλές)

Από Anaconda Prompt `(greek-nlp)` στη ρίζα του repository:

```bat
python scripts\phase15_complete_workflow.py prepare
```

Δημιουργεί `results/phase15_local/phase15_eval78_NOT_FROZEN.json` και `phase15_completion_manifest.json`. Ελέγχει τους 80 υποψήφιους, διαχωρίζει 2 development cases και αποθηκεύει SHA-256. Αν αλλάξει κάποιος κώδικας, το index ή αρχείο του πρωτοκόλλου, η αναφορά σταματά αντί να κάνει σιωπηρή σύγκριση διαφορετικών runs.

## 2. 78 πραγματικές αποκρίσεις (live, CPU)

**Κράτησε ανοιχτό το ήδη ενεργό παράθυρο Uvicorn.** Στο άλλο παράθυρο:

```bat
python scripts\phase15_collect_predictions.py --cases results\phase15_local\phase15_eval78_NOT_FROZEN.json --output results\phase15_local\phase15_eval78_predictions.json --max-requests 78 --confirm-live
```

Οι αποκρίσεις αποθηκεύονται αμέσως μετά από κάθε επιτυχία. Αν η εκτέλεση διακοπεί, εκτέλεσε **την ίδια εντολή** ξανά· το συλλεκτικό script παρακάμπτει όσα έχουν ήδη αποθηκευτεί. Μην τρέχεις δύο collectors ταυτόχρονα ή αλλάζεις μοντέλο, prompts και FAISS κατά τη διάρκεια.

Η εκτέλεση των 78 Qwen requests μπορεί να επιβαρύνει σημαντικά CPU. Δεν έγινε από το παρόν πακέτο και δεν υπάρχει προσποιητή βαθμολογία.

## 3. Αξιολόγηση αποθηκευμένων predictions (offline)

```bat
python scripts\phase15_complete_workflow.py report
```

Παράγει μόνο aggregate report: `phase15_completion_summary_NOT_GOLD.md` και `phase15_final_summary_NOT_GOLD.json`. Όταν υπάρχουν 78/78 predictions, παράγει επίσης `phase15_pilot80_predictions_NOT_GOLD.json`, που ενώνει τις 78 νέες με τις 2 παλιές αποκρίσεις, **χωρίς να τις παρουσιάζει ως held-out**.

## 4. Author review για Faithfulness / Hallucination

```bat
python scripts\phase15_complete_workflow.py review-html
start "" "results\phase15_local\phase15_prediction_author_review.html"
```

Άνοιξε τα snippets, έλεγξε την απάντηση και συμπλήρωσε τα απαιτούμενα πεδία χειροκίνητα. Σε αποχή σημείωσε αν είναι σωστή ή λανθασμένη. Πάτησε **Save explicit review** σε κάθε περίπτωση. Στο τέλος πάτησε **Download reviewed CSV** και μετακίνησε το CSV στο `results/phase15_local/`. Το browser δεν αποθηκεύει αυτομάτως τις καταχωρίσεις στον δίσκο: κατέβαζε ενδιάμεσα το CSV αν χρειάζεται.

Εισαγωγή στο υπάρχον Phase 15 evaluator (χωρίς αυτόματες κρίσεις):

```bat
python -m greek_nlp.evaluation.rag_eval import-review --input results\phase15_local\phase15_pilot80_predictions_NOT_GOLD.json --csv results\phase15_local\phase15_output_author_review.csv --output results\phase15_local\phase15_pilot80_AUTHOR_REVIEWED.json
```

Τελική συνοπτική αναφορά:

```bat
python scripts\phase15_complete_workflow.py reviewed-report --input results\phase15_local\phase15_pilot80_AUTHOR_REVIEWED.json
```

Για ξεχωριστό standard metrics/report από τον evaluator:

```bat
python -m greek_nlp.evaluation.rag_eval score --input results\phase15_local\phase15_pilot80_AUTHOR_REVIEWED.json --output-dir results\phase15_local\scored_pilot80
```

## 5. Τελικοί offline έλεγχοι

```bat
scripts\run_phase15_checks.bat
python -m pytest -q tests\test_phase15_complete_workflow.py
```

## Τι επιτρέπεται να δημοσιευτεί

Προτείνεται μόνο aggregate report, με ρητή σήμανση exploratory/author-reviewed, και με τα limitations. **Όχι** ιδιωτικά ερωτήματα/αποσπάσματα, raw predictions ή ανεπιβεβαίωτα claims. Το Roadmap για πλήρη ανεξάρτητη faithfulness/answerability αξιολόγηση δεν πρέπει να γίνει `[x]` μόνο από το pilot.

## Μη ολοκληρωμένα επιστημονικά gates

1. Ανεξάρτητος, κατά προτίμηση τυφλός αξιολογητής με καταγεγραμμένη συμφωνία επισημάνσεων (αν υπάρχουν 2+ αξιολογητές).
2. Έλεγχος gold unanswerability πέρα από τα περιορισμένα BM25 top-5 και E5 top-8.
3. Πλήρης έλεγχος ανεξαρτησίας από Notebook 10 σε επίπεδο ερώτησης.
4. Διαχωρισμός των 2 development-pilot cases από τις 78 αξιολογήσεις.
5. Faithfulness review των πραγματικών **απαντήσεων**, όχι απλώς των αρχικών ερωτήσεων.
