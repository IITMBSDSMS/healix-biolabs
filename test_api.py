import io
import json
from api_server import app


def run_tests():
    client = app.test_client()

    print("=" * 70)
    print("  HEALIX BACKEND TEST SUITE")
    print("=" * 70)

    # Test 0: Health check
    print("\n[Test 0] Health Check Endpoint")
    res = client.get("/health")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        health = res.get_json()
        print(f"  ✓ API Status: {health.get('status')}")
        print(f"  ✓ Port: {health.get('api_port')}")
        print(f"  ✓ Model Classes: {health.get('model_classes')}")
    else:
        print(f"  ✗ Failed: {res.get_data(as_text=True)}")

    # Test 1: Valid CSV Data
    print("\n[Test 1] Valid CSV Data")
    valid_csv = b"gene1,gene2,gene3,gene4,gene5\n1.2,3.4,5.6,7.8,9.0\n2.1,4.3,6.5,8.7,10.0"
    data = {"file": (io.BytesIO(valid_csv), "valid.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Samples: {result.get('samples')}")
        print(f"  ✓ Accuracy: {result.get('accuracy')}%")
        print(f"  ✓ Predictions shown: {len(result.get('predictions', []))}")
        print(f"  ✓ Genes: {[g['gene'] for g in result.get('genes', [])]}")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    # Test 2: Mixed Text and Numbers
    print("\n[Test 2] Mixed Text and Numbers")
    mixed_csv = b"SampleName,gene1,gene2,gene3,gene4,gene5\nPatientA,1.2,3.4,5.6,7.8,9.0\nPatientB,2.1,4.3,6.5,8.7,10.0"
    data = {"file": (io.BytesIO(mixed_csv), "mixed.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Samples: {result.get('samples')}")
        print(f"  ✓ Summary: {result.get('summary')[:60]}...")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    # Test 3: Text Only (No Numeric Data)
    print("\n[Test 3] Text Only (No Numeric Data) - Expected Error")
    text_csv = b"Name,Type\nSample1,A\nSample2,B"
    data = {"file": (io.BytesIO(text_csv), "text.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 400:
        error = res.get_json()
        print(f"  ✓ Correctly rejected: {error.get('code')}")
    else:
        print(f"  ✗ Should have failed")

    # Test 4: Completely Empty File
    print("\n[Test 4] Empty File - Expected Error")
    empty = b""
    data = {"file": (io.BytesIO(empty), "empty.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 400:
        error = res.get_json()
        print(f"  ✓ Correctly rejected: {error.get('code')}")
    else:
        print(f"  ✗ Should have failed")

    # Test 5: Latin-1 Encoding (non UTF-8)
    print("\n[Test 5] Latin-1 Encoding")
    bad_encoding = "gene1,gene2,gene3,gene4,gene5\n1.0,2.0,3.0,4.0,5.0\n1.5,2.5,3.5,4.5,5.5".encode("latin1")
    data = {"file": (io.BytesIO(bad_encoding), "bad_enc.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Parsed successfully: {result.get('samples')} samples")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    # Test 6: Tab-Separated Values
    print("\n[Test 6] Tab-Separated Values")
    tsv = b"gene1\tgene2\tgene3\tgene4\tgene5\n1.0\t2.0\t3.0\t4.0\t5.0\n1.5\t2.5\t3.5\t4.5\t5.5"
    data = {"file": (io.BytesIO(tsv), "data.tsv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Parsed TSV: {result.get('samples')} samples")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    # Test 7: Data with NaN values
    print("\n[Test 7] Data with Missing Values (NaN)")
    nan_csv = b"gene1,gene2,gene3,gene4,gene5\n1.0,,3.0,4.0,5.0\n,2.5,3.5,,5.5\n1.5,2.5,3.5,4.5,"
    data = {"file": (io.BytesIO(nan_csv), "with_nan.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Handled NaN: {result.get('samples')} samples")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    # Test 8: Large dataset (>25 samples)
    print("\n[Test 8] Large Dataset (50 samples)")
    lines = ["gene1,gene2,gene3,gene4,gene5"]
    for i in range(50):
        lines.append(f"{i+1},{i+2},{i+3},{i+4},{i+5}")
    large_csv = "\n".join(lines).encode()
    data = {"file": (io.BytesIO(large_csv), "large.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Samples processed: {result.get('samples')}")
        print(f"  ✓ Predictions displayed: {len(result.get('predictions', []))} (capped at 25)")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    # Test 9: No file uploaded
    print("\n[Test 9] No File Uploaded - Expected Error")
    res = client.post("/predict", data={}, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 400:
        error = res.get_json()
        print(f"  ✓ Correctly rejected: {error.get('code')}")
    else:
        print(f"  ✗ Should have failed")

    # Test 10: Extreme values
    print("\n[Test 10] Extreme Values")
    extreme = b"gene1,gene2,gene3,gene4,gene5\n1000000,2000000,3000000,4000000,5000000\n0.001,0.002,0.003,0.004,0.005\n1,2,3,4,5"
    data = {"file": (io.BytesIO(extreme), "extreme.csv")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    print(f"  Status: {res.status_code}")
    if res.status_code == 200:
        result = res.get_json()
        print(f"  ✓ Handled extreme values: {result.get('samples')} samples")
        print(f"  ✓ Accuracy: {result.get('accuracy')}%")
    else:
        print(f"  ✗ Failed: {res.get_json()}")

    print("\n" + "=" * 70)
    print("  ✓ ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
