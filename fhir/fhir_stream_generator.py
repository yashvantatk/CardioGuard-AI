"""
HL7 FHIR Release 4 Real-Time RiskAssessment & Observation Bundle Generator
Compatible with SMART on FHIR EHR integrations (Epic / Oracle Cerner).
"""
import json
from datetime import datetime, timezone


def build_fhir_r4_bundle(patient_id: str, patient_name: str, bed: str, risk_prob: float, map_val: int, hr_val: int, lactate_val: float) -> dict:
    return {
        "resourceType": "Bundle",
        "id": f"cardioguard-r4-{patient_id}",
        "meta": {
            "profile": ["http://hl7.org/fhir/R4/StructureDefinition/Bundle"],
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        },
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "name": [{"text": patient_name}],
                    "extension": [{"url": "http://hl7.org/fhir/StructureDefinition/patient-icu-bed", "valueString": bed}]
                }
            },
            {
                "resource": {
                    "resourceType": "RiskAssessment",
                    "id": f"sepsis-risk-{patient_id}",
                    "status": "final",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "prediction": [
                        {
                            "outcome": {"text": "Sepsis-3 Septic Shock Onset"},
                            "probabilityDecimal": round(risk_prob, 3),
                            "whenRange": {"low": {"value": 0, "unit": "h"}, "high": {"value": 6.2, "unit": "h"}}
                        }
                    ]
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"vitals-{patient_id}",
                    "status": "final",
                    "component": [
                        {"code": {"text": "Mean Arterial Pressure"}, "valueQuantity": {"value": map_val, "unit": "mmHg"}},
                        {"code": {"text": "Heart Rate"}, "valueQuantity": {"value": hr_val, "unit": "beats/min"}},
                        {"code": {"text": "Serum Lactate"}, "valueQuantity": {"value": lactate_val, "unit": "mmol/L"}}
                    ]
                }
            }
        ]
    }


if __name__ == "__main__":
    bundle = build_fhir_r4_bundle("icu-04", "Elena Rostova", "BED ICU-04", 0.864, 63, 118, 3.8)
    print(json.dumps(bundle, indent=2))
