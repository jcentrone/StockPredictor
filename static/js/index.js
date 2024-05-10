document.getElementById('tickerForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const ticker = document.getElementById('tickerInput').value;
    fetch(`/predict?ticker=${ticker}`)
        .then(response => response.json())
        .then(data => {
            const trace1 = {
                x: data.dates,  // ensure dates are correctly formatted
                y: data.data,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual Data'
            };
            const trace2 = {
                x: data.dates,
                y: data.baseline_preds,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Baseline Predictions'
            };
            const trace3 = {
                x: data.dates,
                y: data.transformer_preds,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Transformer Predictions'
            };
            const layout = {
                title: 'Stock Price Prediction',
                xaxis: {title: 'Date'},
                yaxis: {title: 'Price'}
            };
            Plotly.newPlot('chart', [trace1, trace2, trace3], layout);
        })
        .catch(error => console.error('Error:', error));
});
