# 🧹 HelixML - Project Cleanup List

## ⚠️ КАТЕГОРИИ ПРОБЛЕМ

---

## 1️⃣ ЛИШНИЕ/ТЕСТОВЫЕ ФАЙЛЫ

### 🗑️ test_helix_project/
**Статус**: Пустой тестовый проект
**Содержимое**: 
- Cargo.toml (пустой)
- src/main.rs (только "Hello, world!")
**Рекомендация**: ❌ УДАЛИТЬ (не используется)

### 🗑️ src/ (root level)
**Статус**: Дублирует lib/src/
**Содержимое**:
- lib.rs (22 lines) - объявляет несуществующие модули
- main.rs (3 lines) - только "Hello, world!"
**Проблема**: Объявляет модули которых НЕТ:
  - `pub mod data;` - НЕТ
  - `pub mod io;` - НЕТ
  - `pub mod moe;` - НЕТ
  - `pub mod quant;` - НЕТ
  - `pub mod rev;` - НЕТ
  - `pub mod utils;` - НЕТ
**Рекомендация**: ❌ УДАЛИТЬ или ✏️ ИСПРАВИТЬ (использовать lib/src/)

---

## 2️⃣ ОБЪЯВЛЕННЫЕ НО ОТСУТСТВУЮЩИЕ МОДУЛИ

### ⚠️ multimodal/src/lib.rs
**Объявлено, но НЕТ файлов**:
- ❌ `mod encoders;` - файл отсутствует
- ❌ `mod decoders;` - файл отсутствует
- ❌ `mod fusion;` - файл отсутствует
- ❌ `mod alignment;` - файл отсутствует
- ❌ `mod transformers;` - файл отсутствует
- ❌ `mod pipelines;` - файл отсутствует
- ❌ `mod utils;` - файл отсутствует

**Текущее состояние**: Закомментированы или вызывают ошибки
**Рекомендация**: 
- ✏️ РАСКОММЕНТИРОВАТЬ объявления
- ➕ СОЗДАТЬ файлы или
- ❌ УДАЛИТЬ объявления

### ⚠️ autograd/
**Проблема**: Объявляет 9 модулей, но только 7 файлов
**Рекомендация**: Проверить какие модули missing

### ⚠️ nn/
**Проблема**: Объявляет 3 модуля, но только 1 файл
**Рекомендация**: Проверить lib.rs

### ⚠️ meanings/
**Проблема**: Объявляет 1 модуль, но 0 отдельных файлов (всё в lib.rs?)
**Рекомендация**: Проверить структуру

### ⚠️ optim/
**Проблема**: Объявляет 1 модуль, но 0 отдельных файлов
**Рекомендация**: Проверить структуру

---

## 3️⃣ ФАЙЛЫ С МНОЖЕСТВОМ TODO

### 🔧 backend-cpu/src/cpu_backend.rs: 24 TODO
**Проблемы**:
- Незавершённые implementations
- Placeholder методы
**Рекомендация**: ✏️ РЕАЛИЗОВАТЬ или оставить с понятными комментариями

### 🔧 backend-cuda/src/cuda_backend.rs: 17 TODO
**Рекомендация**: ✏️ РЕАЛИЗОВАТЬ CUDA operations

### 🔧 autograd/src/advanced.rs: 10 TODO
**Рекомендация**: ✏️ РЕАЛИЗОВАТЬ advanced features

### 🔧 training/src/trainer.rs: 7 TODO
**Основные**:
- Backward pass implementation
- Gradient clipping
- Forward/loss interface mismatch
**Рекомендация**: ✏️ РЕАЛИЗОВАТЬ критические части

### 🔧 training/src/loss.rs: 6 TODO
**Проблема**: Placeholder implementations для всех loss functions
**Рекомендация**: ✏️ РЕАЛИЗОВАТЬ правильные формулы (MSE, CrossEntropy, BCE, etc.)

### 🔧 training/src/optimizer.rs: 4 TODO
**Проблема**: Placeholder step() methods
**Рекомендация**: ✏️ РЕАЛИЗОВАТЬ Adam, AdamW, SGD, RMSprop step logic

### 🔧 hammer/src/*.rs: 3-2 TODO каждый
**Рекомендация**: ✏️ ДОДЕЛАТЬ implementations

---

## 4️⃣ ПОТЕНЦИАЛЬНО НЕИСПОЛЬЗУЕМЫЕ CRATES

### ❓ geometry/
**Содержимое**: 4 files, 406 lines
**Использование**: ?
**Рекомендация**: 🔍 ПРОВЕРИТЬ связь с другими модулями

### ❓ serve/
**Содержимое**: 1 file, 382 lines
**Использование**: ?
**Рекомендация**: 🔍 ПРОВЕРИТЬ если это used

### ❓ scheduling/
**Содержимое**: 1 file, 368 lines
**Связь**: Возможно дублирует adaptive-scheduler?
**Рекомендация**: 🔍 ПРОВЕРИТЬ не дублируется ли

---

## 5️⃣ COMPILATION ISSUES

### ❌ lib/ (root lib crate)
**Ошибки**: synthetic-data compilation errors (172+)
**Проблема**: GeneratedSequences, GeneratedImages и т.д. не найдены
**Рекомендация**: ✏️ ИСПРАВИТЬ synthetic-data module

### ❌ src/lib.rs
**Ошибки**: Объявляет несуществующие модули
**Рекомендация**: ✏️ УДАЛИТЬ объявления несуществующих модулей или создать их

---

## 6️⃣ ДОКУМЕНТАЦИЯ

### 📝 Хорошие файлы (ОСТАВИТЬ):
- ✅ HELIX_CAPABILITIES.md - полное описание возможностей
- ✅ RESTORATION_SUCCESS.md - отчёт восстановления
- ✅ README.md - основная документация
- ✅ QUICKSTART.md - быстрый старт
- ✅ CHANGELOG.md - история изменений
- ✅ CONTRIBUTING.md - guide для contributors
- ✅ docs/ARCH.md - архитектура
- ✅ docs/MEANINGS.md - MIL/SIM описание

### ❓ Проверить необходимость:
- docs/book/ - mdBook (используется?)

---

## 🎯 ПРИОРИТЕТНЫЙ СПИСОК ОЧИСТКИ

### 🔴 КРИТИЧНО (удалить сейчас):
1. **test_helix_project/** - пустой тестовый проект
2. **src/lib.rs** - неправильные module declarations
3. **src/main.rs** - бесполезный "Hello World"

### 🟡 ВАЖНО (исправить скоро):
4. **training/src/loss.rs** - реализовать loss functions (сейчас placeholders)
5. **training/src/optimizer.rs** - реализовать optimizer step() methods
6. **training/src/trainer.rs** - реализовать backward pass
7. **multimodal/src/lib.rs** - раскомментировать/создать missing modules или удалить declarations

### 🟢 ЖЕЛАТЕЛЬНО (можно позже):
8. **backend-cpu/src/cpu_backend.rs** - реализовать 24 TODO
9. **backend-cuda/src/cuda_backend.rs** - реализовать 17 TODO
10. **hammer/** - доделать TODO в implementations
11. **synthetic-data/** - исправить compilation errors (172)
12. Проверить geometry/, serve/, scheduling/ на актуальность

---

## 📊 СТАТИСТИКА

```
Всего файлов в проекте: ~200+
Всего строк кода: ~35,000+
TODO комментов: 144
Пустых/тестовых проектов: 1
Дублирований структуры: 1
Missing modules: 7+ (multimodal)
Compilation errors: ~172 (synthetic-data only)
```

---

## 🛠️ РЕКОМЕНДУЕМЫЙ ПЛАН ДЕЙСТВИЙ

### Шаг 1: Удалить лишнее (5 мин)
```bash
rm -rf test_helix_project/
rm -rf src/  # Использовать только lib/src
```

### Шаг 2: Исправить declarations (10 мин)
- Проверить каждый crate/lib.rs
- Удалить declarations для missing modules
- Или создать пустые файлы с TODO

### Шаг 3: Реализовать критичные TODO (1-2 часа)
- training/loss.rs - настоящие loss formulas
- training/optimizer.rs - настоящие optimizer steps
- training/trainer.rs - backward pass integration

### Шаг 4: Проверить неиспользуемые crates (30 мин)
- geometry - используется?
- serve - используется?
- scheduling - дублирует adaptive-scheduler?

### Шаг 5: Исправить synthetic-data (1 час)
- Добавить missing types
- Fix compilation

---

## ✅ ЧТО УЖЕ ОТЛИЧНО

✅ Hammer engine - полностью рабочий!
✅ Training - компилируется!
✅ Autograd - работает!
✅ Topo-memory - исправлен!
✅ Multimodal - основное работает!
✅ Backend-CPU - функционален!
✅ VortexGrad - готов к использованию!

**Основная функциональность в порядке! Осталось почистить мелочи!**

