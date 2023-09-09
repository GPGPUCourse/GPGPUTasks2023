// Убедитесь что название PR соответствует шаблону:
// Task0N <Имя> <Фамилия> <Аффиляция>
// И проверьте что обе ветки PR (отправляемая из вашего форкнутого репозитория и та в которую вы отправляете PR) называются одинаково - task0N

// Впишите сюда (между pre и /pre тэгами) вывод тестирования на вашем компьютере:

<details><summary>Локальный вывод</summary><p>

<pre>
$ ./enumDevices
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: Intel(R) CPU Runtime for OpenCL(TM) Applications
    Platform vendor: Intel(R) Corporation
    Number of platform devices: 1
    Device #1/1
        Device name: AMD Ryzen 9 3900X 12-Core Processor            
        Device type: 2
        Device memory size: 32768
        Device platform: FULL_PROFILE
</pre>

</p></details>

// Затем создайте PR, должна начать выполняться автоматическиая сборка на Github CI (Github Actions) - рядом с коммитом в PR появится оранжевый шарик (сборка в процессе),
// который потом станет зеленой галкой (прошло успешно) или красным крестиком (что-то пошло не так).
// Затем откройте PR на редактирование чтобы добавить в описание (тоже между pre и /pre тэгами) вывод тестирования на Github CI:
// Чтобы его найти - надо нажать на зеленую галочку или красный крестик рядом с вашим коммитов в рамках PR.
// P.S. В случае если Github CIсборка не запустилась - попробуйте через десять минут или через час добавить фиктивный коммит (например добавив где-то пробел).

<details><summary>Вывод Github CI</summary><p>

<pre>
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: Intel(R) CPU Runtime for OpenCL(TM) Applications
    Platform vendor: Intel(R) Corporation
    Number of platform devices: 1
    Device #1/1
        Device name: Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz
        Device type: 2
        Device memory size: 32768
        Device platform: FULL_PROFILE
</pre>

</p></details>
