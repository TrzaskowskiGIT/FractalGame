# FractalGame

Uruchomienie

UWAGA: Program wymaga systemu Windows oraz karty graficznej NVIDIA obsługującej CUDA.

Aby uruchomić aplikację, należy upewnić się, że na komputerze jest zainstalowane środowisko CUDA w wersji 12.9.2 lub kompatybilnej z używaną kartą graficzną.

GPU Fractal Explorer.exe

Program korzysta z algorytmu przeznaczonego dla języka CUDA do renderowania fraktali, dlatego na komputerach bez karty NVIDIA lub bez poprawnie zainstalowanej CUDA aplikacja nie będzie działać poprawnie.



Sterowanie

Akcja

Sterowanie

otwarcie menu

ESC 

Poruszanie widokiem

W, A, S, D

Zwiększenie liczby iteracji

E

Zmniejszenie liczby iteracji

Q

Wyśrodkowanie widoku

Lewy przycisk myszy

Przybliżenie

Scroll w górę

Oddalenie

Scroll w dół

Zmiana palety kolorów

R / F

Eksport obrazu PNG

P

Zmiana parametrów fraktala Mandelbrota/Julii

I, J, K, L

Obsługa menu

ESC, kliknięcie myszą





Opis kluczowych funkcjonalności

1\. Renderowanie GPU — CuPy + CUDA

Aplikacja renderuje fraktale z użyciem karty graficznej NVIDIA. Obliczenia wykonywane są równolegle na GPU, co pozwala na szybkie generowanie obrazu nawet przy dużej liczbie iteracji. Program wykorzystuje własny kernel CUDA. W renderowaniu zastosowano również arytmetykę typu double-double, która poprawia dokładność przy dużych przybliżeniach.

2\. Obsługa wielu fraktali

Program pozwala wyświetlać kilka typów fraktali:

Mandelbrot,

Julia Set,

Burning Ship.

Użytkownik może przełączać typ fraktala w menu oraz korzystać z presetów dla fraktali Mandelbrota i Julii.

3\. Dynamiczne przybliżanie i poruszanie kamery

Aplikacja umożliwia płynne przybliżanie i oddalanie widoku za pomocą scrolla myszy. Widok można przesuwać klawiszami W, A, S, D, a także wyśrodkować kamerę na wybranym punkcie przez kliknięcie lewym przyciskiem myszy.

4\. Kolorowanie fraktala

Program posiada kilka palet kolorów, które można zmieniać w czasie działania aplikacji. Kolor pikseli zależy od liczby iteracji potrzebnych do określenia zachowania punktu. Dzięki temu obraz jest czytelny i atrakcyjny wizualnie.

5\. Adaptacyjna dokładność

Użytkownik może zmieniać liczbę iteracji podczas działania programu. Większa liczba iteracji pozwala uzyskać dokładniejszy i bardziej szczegółowy obraz, ale może obniżyć płynność renderowania. Mniejsza liczba iteracji działa szybciej, ale daje mniej dokładny wynik.

6\. Eksport obrazów i animacji

Aplikacja umożliwia eksport aktualnego widoku jako obrazu PNG oraz wygenerowanie animacji GIF z przybliżeniem fraktala. Pliki eksportowane są do osobnych folderów:

exports/pictures — obrazy PNG,

exports/gifs — animacje GIF.

7\. Menu ustawień

Program posiada menu, w którym można zmieniać m.in.:

typ fraktala,

paletę kolorów,

rozdzielczość okna,

rozdzielczość eksportu PNG,

rozdzielczość eksportu GIF,

ustawienia audio,

widoczność informacji o wydajności.

8\. Benchmark GPU vs CPU

W aplikacji znajduje się test wydajności porównujący renderowanie na GPU i CPU. Wynik benchmarku pokazuje średni czas renderowania oraz przyspieszenie uzyskane dzięki GPU.

9\. Samouczek i osiągnięcia

Program zawiera interaktywny samouczek, który prowadzi użytkownika przez podstawowe akcje, takie jak ruch, zoom, zmiana iteracji i zmiana palety. Aplikacja posiada też system osiągnięć, np. za ukończenie tutoriala lub wykonanie eksport.



Spis błędów

Aplikacja nie uruchamia się poprawnie, gdy CUDA nie jest zainstalowana, często wymuszając restart komputera.

Nietypowe zachowania:

Zmiana rozdzielczości okna na mniejsze od podstawowego nie przesuwa okna na środek ekranu, więc program należy zrestartować aby przesunąć okno.

na najmniejszym możliwym rozmiarze okna gry część menu jest ucięta.

Pierwsze uruchomienie programu zajmuje trochę czasu i nie ma niczego co by sugerowało, że program się uruchamia.

Gra jest ciężka, po uruchomieniu jej zalecane jest zmniejszenie rozmiaru okna na 2-gie najmniejsze oraz nie zwiększanie liczby iteracji na więcej niż 200 (startowe to 100).

Według testów gra wprowadza osobę grającą i potencjalnie osoby oglądające w stan niepokoju.



