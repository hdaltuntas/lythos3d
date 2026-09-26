# Lythos 3D kullanım kılavuzu

Bu kılavuz ilk sahanızı çizip analiz etmeyi adım adım anlatır. Formülasyon ve
doğrulamalar için [theory.md](theory.md), İngilizce özet için
[README](../README.md).

## 1. Kurulum

Linux'ta (macOS benzer):

```bash
cd lythos3d
python3 -m venv .venv
source .venv/bin/activate          # fish kabuğunda: source .venv/bin/activate.fish
pip install -e ".[dev]"
```

Bu komut numpy, scipy, pypardiso (hızlı çözücü), gmsh (ağ oluşturucu) ve
pytest'i kurar. Ekranı olmayan bir sunucuda gmsh birkaç sistem kütüphanesi
daha ister:

```bash
sudo apt install libglu1-mesa libxcursor1 libxinerama1 libxft2
```

Kontrol için `python main.py info` komutunu çalıştırın. Çıktıda **PARDISO**
görünmelidir. Görünmüyorsa `pip install pypardiso` ile kurun; PARDISO
olmadan program yavaş SuperLU çözücüsüne düşer ve gerçekçi bir saha saatler
sürebilir.

## 2. Arayüzü açmak

```bash
python main.py
```

Tarayıcıda `http://127.0.0.1:8778/` açılır (PyCharm'da parametre vermeden
Run'a basmak aynı şeyi yapar). Kapatmak için konsolda Ctrl+C. Arayüz
tarayıcının diline göre Türkçe ya da İngilizce açılır; üstteki menüden
değiştirilebilir.

İlk kez deniyorsanız **Örnekler** menüsünden "walled pit"i seçin: dipli
katmanlı zeminde diyafram duvarlı, destekli bir kazı.

## 3. Sahayı çizmek

Üst çubuktaki araçlar:

| Araç | Ne yapar |
| --- | --- |
| **Seç** | Öğeye tıklayıp düzenleme, köşeleri sürükleme, boş alanı sürükleyerek kaydırma; tekerlekle yakınlaşma. Delete seçili öğeyi siler. |
| **Sınır** | Modelin plan sınırını dikdörtgen olarak sürükleyin. |
| **Sondaj** | Tıklayıp yerleştirin; zemin katmanlarını sağdaki **Öğe** sekmesinde, her satıra "zemin_adı üst_kotu" olarak girin. |
| **Kazı** | Köşelere tıklayın, çift tıklama ya da Enter ile bitirin. Kazı kademelerinin kotlarını (ör. `-1.5, -3`) ve susuzlaştırmayı Öğe sekmesinde girin. |
| **Duvar** | Duvar boyunca tıklayın; kazının çevresini kapatmak için ilk noktaya tekrar tıklayın. Uç kotu, kesit (E, t) ve arayüz (R) Öğe sekmesinde. |
| **Dolgu** | Set ya da platform poligonu; kotları yükselen sırayla (ör. `1.5, 3`). |
| **Yük** | Yüklü alan poligonu; q (kPa, plan alanı başına) ve isteğe bağlı yatay yük. |
| **Ankraj** | Önce baş, sonra uzak uç; kotlar, EA ve kilitleme kuvveti Öğe sekmesinde (destek için eksi). |
| **Kazık** | Tıklayıp yerleştirin; baş ve uç kotu, çap, kapasiteler Öğe sekmesinde. |

Bir DXF planınız varsa **DXF altlık…** ile yükleyin. Kazı, duvar, dolgu ya da
yük aracı seçiliyken bir DXF çizgisine tıklamak onu doğrudan alır.

**Zeminler** sekmesinde her zeminin modeli ve parametreleri girilir (birimler
kPa, kN/m³, derece):

- **mohr-coulomb**: E, ν, γ, γsat, c, φ, ψ.
- **stress-dependent-mohr-coulomb**: E'yi 100 kPa'daki modül olarak alır; E_ur
  boşaltma modülüdür (boş bırakılırsa 3E), m üssüdür (boşsa 0.5). Kazı
  tabanının kabarmasını ve duvar arkasındaki deformasyonları Mohr-Coulomb'dan
  daha gerçekçi verir.
- **linear-elastic**: yalnızca E, ν, γ.
- **drainage**: "undrained" seçilirse zemin yüklemede aşırı boşluk suyu basıncı
  üretir (drenajsız A; φ = 0 ve c = su ile drenajsız B).
- **k**: geçirgenlik; yalnızca sızma ve konsolidasyonda kullanılır.

**Saha** sekmesinde taban kotu, ağ boyutu ve su seviyesi girilir. Su seviyesi
boş bırakılırsa zemin kuru kabul edilir. "Sızmayı çöz" işaretlenirse boşluk
suyu basıncı hidrostatik varsayılmaz, akış çözülür. Pompalanan bir kazıda
suyun duvarın altından dolaşmasını görmek için bunu kullanın.

**3B görünüm** modeli ağ oluşturmadan önce gösterir: sondajlardan
interpole edilen katmanlar, kazılar, duvarlar, dolgular, yükler, su tablası.
"içini göster" yakın yan yüzleri keser.

**Kontrol** sekmesi eksik ya da hatalı girişleri listeler.

## 4. Analiz

Üst çubukta:

- **Ağ**: kaba, normal ya da ince. İlk denemede kaba ağ kullanın; birkaç kat
  daha hızlıdır.
- **Güvenlik sayısı**: varsayılan olarak kapalıdır. Mukavemet azaltma yöntemi
  modeli birçok kez yeniden çözer ve inşaat aşamalarının birkaç katı sürer
  (orta boy bir sahada onlarca dakika). Gerektiğinde önce "hızlı (±0.03)"
  seçeneğini kullanın.
- **Analizi çalıştır**: aşamalar Lythos 3D tarafından çizilenden kurulur:
  başlangıç gerilmeleri, duvar ve kazıklar, dolgular, yükler, her kazı
  kademesi (ankrajlarıyla), sonra istenirse güvenlik sayısı.

**Analiz** sekmesi ilerlemeyi aşama aşama gösterir; hangi aşamada, yükün
yüzde kaçında, kaçıncı iterasyonda olunduğunu görürsünüz. **Durdur** analizi
keser.

## 5. Sonuçlar

Analiz bitince **Sonuçlar** düğmesi sonucu aynı ekranda 3B gösterir: aşama ve
büyüklük seçimi (yer değiştirme, gerilmeler, plastik şekil değiştirme, boşluk
suyu basıncı, hidrolik yük), deformasyon ölçeği, kesit düzlemi. **Model**
çizime döner.

**Rapor** tam raporu yeni sekmede açar: malzemeler, aşama tablosu (yer
değiştirme, güvenlik sayısı, duvar momentleri, ankraj kuvvetleri, sızma
debileri), grafikler (güvenlik sayısı araması, konsolidasyon, duvar moment
zarfı, kazık eksenel kuvveti) ve 3B görüntüleyici. Rapor tek bir HTML
dosyasıdır ve internetsiz açılır.

Her analizin dosyaları `~/lythos3d_runs/<tarih>-<saha adı>/` altında
saklanır: `site.json`, `report.html`, `viewer.html` ve ParaView için
`stage*.vtu`, `plates*.vtu`.

## 6. Komut satırı

```bash
python main.py site-example -o site.json          # örnek saha dosyası
python main.py run site.json -o run --lang tr     # analiz; rapor run/report.html
python main.py run site.json -o run --no-fos      # güvenlik sayısı olmadan
python main.py dxf plan.dxf                       # bir DXF'in katmanlarını listele
python main.py editor -o editor.html              # editörü tek dosya olarak yaz
python main.py --help                             # tüm komutlar
```

Site dosyasında her poligon ya da duvar yolu bir DXF katmanına referans
verebilir: `"polygon": {"dxf": "plan.dxf", "layer": "PIT"}`.

## 7. Bilinmesi gerekenler

- **Hidrostatik su indirme**: kazı içinde su indirilip kenarında duvar yoksa,
  basınç sıçraması zemine gerçekte olmayan bir yük olarak biner. Program bunu
  uyarır; duvar ekleyin ya da sızmayı çözün.
- **Güvenlik sayısı süresi**: yukarıda anlatıldığı gibi uzundur. Kaba ağ ve
  hızlı seçenekle başlayın.
- **Gerilmeye bağlı model** Hardening Soil'in elastik iki etkisini
  (sıkışmaya bağlı rijitlik, boşaltma/yeniden yükleme modülü) içerir,
  pekleşen plastisitesini ve kapağını içermez.
- **Konsolidasyon** aşamasında zaman birimi geçirgenliğin birimiyle aynıdır
  (k m/gün ise zaman gün).

## 8. Sık karşılaşılan sorunlar

| Belirti | Çözüm |
| --- | --- |
| `No module named numpy` | Sanal ortamı etkinleştirin ya da `pip install -e ".[dev]"` |
| gmsh "libGLU" hatası | `sudo apt install libglu1-mesa libxcursor1 libxinerama1 libxft2` |
| Analiz çok yavaş, Analiz sekmesinde SuperLU uyarısı | `pip install pypardiso` |
| Güvenlik sayısı çok uzun sürüyor | Kaba ağ, hızlı seçenek; gerekirse Durdur |
| "a stage did not converge" | Zemin o aşamada göçüyor olabilir; güvenlik sayısı olmadan inşaat aşamalarına bakın, yük ya da kazı derinliğini gözden geçirin |
