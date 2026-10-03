import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

const KEY = "sksurrogate_task";

interface TaskContextValue {
  task: string;
  setTask: (t: string) => void;
}

const TaskContext = createContext<TaskContextValue>({ task: "", setTask: () => {} });

export function TaskProvider({ children }: { children: ReactNode }) {
  const [task, setTaskState] = useState<string>(() => localStorage.getItem(KEY) ?? "");
  const setTask = (t: string) => {
    setTaskState(t);
    if (t) localStorage.setItem(KEY, t);
    else localStorage.removeItem(KEY);
  };
  return <TaskContext.Provider value={{ task, setTask }}>{children}</TaskContext.Provider>;
}

export function useTask(): TaskContextValue {
  return useContext(TaskContext);
}
