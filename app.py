
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat model Random Forest
model = joblib.load('random_forest_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Lima fitur final
FITUR_FINAL = ['17', '22', '23', '25', 'COURSE ID']

if feature_columns != FITUR_FINAL:
    raise ValueError(
        f"Fitur model tidak sesuai: {feature_columns}. "
        f"Seharusnya: {FITUR_FINAL}"
    )

# Keterangan hasil prediksi
GRADE_LABELS = {
    0: 'Fail (Tidak Lulus)',
    1: 'DD',
    2: 'DC',
    3: 'CC',
    4: 'CB',
    5: 'BB',
    6: 'BA',
    7: 'AA'
}


@app.route('/')
def index():
    return render_template(
        'index.html',
        prediction=None,
        grade_label=None,
        error=None
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {}

        for fitur in feature_columns:
            nilai = request.form.get(fitur)

            if not nilai:
                raise ValueError('Semua kolom wajib diisi.')

            data[fitur] = int(nilai)

        input_data = pd.DataFrame(
            [data],
            columns=feature_columns
        )

        hasil = int(model.predict(input_data)[0])
        grade_label = GRADE_LABELS.get(
            hasil,
            'Kategori tidak dikenal'
        )

        return render_template(
            'index.html',
            prediction=hasil,
            grade_label=grade_label,
            error=None
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction=None,
            grade_label=None,
            error=str(e)
        )


if __name__ == '__main__':
    app.run(debug=True)
