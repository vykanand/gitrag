async function searchCode() {
    const query = document.getElementById('query').value;
    const response = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });
    const results = await response.json();
    displayResults(results);
}

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    results.forEach(result => {
        const div = document.createElement('div');
        div.classList.add('result');
        div.innerHTML = `
            <div class="filename">${result[0]} (${result[1]}) - Distance: ${result[3]}</div>
            <pre>${result[2]}</pre>
        `;
        resultsDiv.appendChild(div);
    });
}
