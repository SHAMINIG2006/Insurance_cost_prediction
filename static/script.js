// AJAX predict to /api/predict
document.getElementById('ajaxBtn').addEventListener('click', function(){
  const form = document.getElementById('predictForm');
  const fd = new FormData(form);
  const data = {};
  fd.forEach((v,k)=> data[k]=v);

  fetch('/api/predict', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(data)
  }).then(res => res.json())
    .then(j => {
      document.getElementById('amount').textContent = j.prediction;
      document.getElementById('resultCard').classList.remove('hidden');
      window.scrollTo({top: document.getElementById('resultCard').offsetTop - 20, behavior: 'smooth'});
    }).catch(err => alert('Error: ' + err));
});
